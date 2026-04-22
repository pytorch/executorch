from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    ListType,
    NativeFunction,
    OptionalType,
    Type,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.api.types import Binding, CType, NamedCType


connector = "\n\t"


# Return unboxing function name for a NativeFunction
def name(f: NativeFunction) -> str:
    return f.func.name.unambiguous_name()


@dataclass(frozen=True)
class Unboxing:
    """
    Takes a sequence of Bindings and unbox EValues to these Bindings. Return generated code that performs correct unboxing.
    A sample generated code (abbreviated to one arg for readability):
    // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    void mul_out(KernelRuntimeContext& context, Span<EValue*> stack) {
        EValue& self = *stack[0];
        // ... other args ...
        auto self_base_res = self.tryTo<torch::executor::Tensor>();
        if (!self_base_res.ok()) {
          ::executorch::runtime::internal::kernel_arg_fail(
              context, self_base_res.error(), __func__, "self",
              static_cast<uint8_t>(self.tag));
          return;
        }
        const torch::executor::Tensor& self_base = self_base_res.get();
        // ... other unpacks ...
        EXECUTORCH_SCOPE_PROF("native_call_mul.out");
        torch::executor::mul_outf(self_base, other_base, out_base);
    }
    """

    # this is a callable that converts a JIT argument, into its C++ type.
    # Translates (type, mutability, binds) to NamedCType. E.g., torchgen.api.cpp.argumenttype_type.
    argument_type_gen: Callable[
        ...,
        NamedCType,
    ]

    # Convert all the arguments in a NativeFunction to C++ code
    def convert_arguments(
        self, args: Sequence[Binding]
    ) -> tuple[list[Binding], list[str]]:
        code_list = [f"EValue& {args[i].name} = *stack[{i}];" for i in range(len(args))]
        binding_list = []
        for arg in args:
            # expecting only Argument
            if not isinstance(arg.argument, Argument):
                raise Exception(  # noqa: TRY002
                    f"Unexpected argument type, expecting `Argument` but got {arg}"
                )
            argument: Argument = arg.argument
            unboxed_name, _, code, decl = self.argumenttype_evalue_convert(
                argument.type, argument.name, mutable=argument.is_write
            )
            code_list.extend(decl)
            code_list.extend(code)
            binding_list.append(arg.with_name(unboxed_name))
        return binding_list, code_list

    def argumenttype_evalue_convert(
        self, t: Type, arg_name: str, *, mutable: bool = False
    ) -> tuple[str, CType, list[str], list[str]]:
        """
        Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
        (1) the C++ code necessary to unbox the argument
        (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
        :param t: a `Type` of an argument
        :param arg_name: argument name
        :param mutable: boolean for whether this argument type is mutable
        :return: unboxed result
        """
        ctype = self.argument_type_gen(t, mutable=mutable, binds=arg_name).type

        if isinstance(t, BaseType):
            out_name = f"{arg_name}_base"
            code, decl = self._gen_code_base_type(
                arg_name=arg_name, out_name=out_name, ctype=ctype
            )
        elif isinstance(t, OptionalType):
            out_name = f"{arg_name}_opt_out"
            code, decl = self._gen_code_optional_type(
                arg_name=arg_name, out_name=out_name, t=t, ctype=ctype
            )
        elif isinstance(t, ListType):
            out_name = f"{arg_name}_list_out"
            code, decl = self._gen_code_list_type(
                arg_name=arg_name, out_name=out_name, t=t, ctype=ctype
            )
        else:
            raise Exception(  # noqa: TRY002
                f"Cannot handle type {t}. arg_name: {arg_name}"
            )  # noqa: TRY002
        return out_name, ctype, code, decl

    def _gen_code_base_type(
        self, arg_name: str, out_name: str, ctype: CType
    ) -> tuple[list[str], list[str]]:
        # Use tryTo<T>() with a shared cold fail helper so every wrapper
        # logs a consistent diagnostic and propagates the error via
        # KernelRuntimeContext::fail() rather than aborting.
        res_name = f"{out_name}_res"
        return [
            f"auto {res_name} = {arg_name}.tryTo<{ctype.cpp_type(strip_ref=True)}>();",
            f"if (!{res_name}.ok()) {{",
            "  ::executorch::runtime::internal::kernel_arg_fail(",
            f'      context, {res_name}.error(), __func__, "{arg_name}",',
            f"      static_cast<uint8_t>({arg_name}.tag));",
            "  return;",
            "}",
            f"{ctype.cpp_type()} {out_name} = {res_name}.get();",
        ], []

    def _gen_code_optional_type(
        self, arg_name: str, out_name: str, t: OptionalType, ctype: CType
    ) -> tuple[list[str], list[str]]:
        in_name = f"{arg_name}_opt_in"
        res_name, base_type, res_code, decl = self.argumenttype_evalue_convert(
            t.elem, in_name
        )
        # Use tryToOptional<T>() with the shared fail helper (see
        # _gen_code_base_type).
        opt_res_name = f"{out_name}_res"
        return (
            f"""
    auto {opt_res_name} = {arg_name}.tryToOptional<{base_type.cpp_type(strip_ref=True)}>();
    if (!{opt_res_name}.ok()) {{
      ::executorch::runtime::internal::kernel_arg_fail(
          context, {opt_res_name}.error(), __func__, "{arg_name}",
          static_cast<uint8_t>({arg_name}.tag));
      return;
    }}
    auto {out_name} = std::move({opt_res_name}.get());
            """.split("\n"),
            decl,
        )

    def _gen_code_list_type(
        self, arg_name: str, out_name: str, t: ListType, ctype: CType
    ) -> tuple[list[str], list[str]]:
        in_name = f"{arg_name}_list_in"
        elem_name = f"{arg_name}_elem"
        code = []
        res_name, res_ctype, res_code, decl = self.argumenttype_evalue_convert(
            t.elem, elem_name
        )

        # Each branch uses the Result-returning tryToXList() accessor and
        # routes errors through the shared kernel_arg_fail helper; see
        # _gen_code_base_type for the rationale.
        res_name_list = f"{out_name}_res"

        def _fail_block(res: str) -> str:
            # Cold fail path: log + context.fail() via the shared helper.
            return (
                f"if (!{res}.ok()) {{\n"
                f"      ::executorch::runtime::internal::kernel_arg_fail(\n"
                f'          context, {res}.error(), __func__, "{arg_name}",\n'
                f"          static_cast<uint8_t>({arg_name}.tag));\n"
                f"      return;\n"
                f"    }}"
            )

        if isinstance(t.elem, BaseType) and t.elem.name == BaseTy.Tensor:
            code.extend(f"""
    auto {res_name_list} = {arg_name}.tryToTensorList();
    {_fail_block(res_name_list)}
    auto {out_name} = {res_name_list}.get();
                """.split("\n"))
        elif isinstance(t.elem, BaseType) and (
            t.elem.name == BaseTy.int or t.elem.name == BaseTy.SymInt
        ):
            code.extend(f"""
    auto {res_name_list} = {arg_name}.tryToIntList();
    {_fail_block(res_name_list)}
    auto {out_name} = {res_name_list}.get();
                """.split("\n"))
        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.float:
            code.extend(f"""
    auto {res_name_list} = {arg_name}.tryToDoubleList();
    {_fail_block(res_name_list)}
    auto {out_name} = {res_name_list}.get();
                """.split("\n"))
        elif isinstance(t.elem, BaseType) and t.elem.name == BaseTy.bool:
            # handle list type with size, e.g., bool[4]
            code.extend(f"""
#ifdef USE_ATEN_LIB
std::array<bool, {t.size}> {out_name};
auto {in_name}_res = {arg_name}.tryToBoolList();
{_fail_block(in_name + "_res")}
auto {in_name} = {in_name}_res.get();
size_t _i = 0;
for (auto {elem_name}: {in_name}) {{
    {out_name}[_i++] = {elem_name};
}}
#else
auto {res_name_list} = {arg_name}.tryToBoolList();
{_fail_block(res_name_list)}
auto {out_name} = {res_name_list}.get();
#endif
                """.split("\n"))
        # pytorch codegen:
        # we have to use c10::List for optional element. e.g., Tensor?[] -> c10::List<::std::optional<at::Tensor>>
        elif (
            isinstance(t.elem, OptionalType)
            and isinstance(t.elem.elem, BaseType)
            and t.elem.elem.name == BaseTy.Tensor
        ):
            code.extend(f"""
#ifdef USE_ATEN_LIB
auto {in_name}_res = {arg_name}.tryToListOptionalTensor();
{_fail_block(in_name + "_res")}
auto {in_name} = {in_name}_res.get();
c10::List<::std::optional<at::Tensor>> {out_name};
for (auto {elem_name}: {in_name}) {{
    {out_name}.push_back({elem_name});
}}
#else
auto {res_name_list} = {arg_name}.tryToListOptionalTensor();
{_fail_block(res_name_list)}
auto {out_name} = {res_name_list}.get();
#endif
                """.split("\n"))
        else:
            # use ArrayRef as default.
            vec_name = arg_name + "_vec"
            # need to bring vector instantiation out of scope so that ArrayRef has valid data
            decl.append(
                f"std::vector<{res_ctype.cpp_type(strip_ref=True)}> {vec_name};"
            )
            code.extend(f"""
    for (EValue {elem_name}: {in_name}) {{
        {connector.join(res_code)}
        {vec_name}.push_back({res_name});
    }}
    {ctype.cpp_type(strip_ref=True)} {out_name}({vec_name});
                """.split("\n"))
        return code, decl
