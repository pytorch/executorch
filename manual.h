namespace torch {
namespace executor {
    void manual_override();
    void digant_add_out(torch::executor::KernelRuntimeContext & context, EValue** stack);
}}
