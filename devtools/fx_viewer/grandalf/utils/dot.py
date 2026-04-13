# This code is part of Grandalf
# Copyright (C) 2008 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Axel Tillequin

try:
    import ply.lex as lex
    import ply.yacc as yacc

    _has_ply = True
except ImportError:
    _has_ply = False

__all__ = ["_has_ply", "Dot"]

# ------------------------------------------------------------------------------
# LALR(1) parser for Graphviz dot file format.
class Dot:

    _reserved = (
        "strict",
        "graph",
        "digraph",
        "subgraph",
        "node",
        "edge",
    )
    _tokens = ("regulars", "string", "html", "comment",) + _reserved

    _literals = [",", ";", "-", ">", "=", ":", "[", "]", "{", "}"]

    class Lexer(object):
        def __init__(self):
            self.whitespace = "\0\t\n\f\r "
            self.reserved = Dot._reserved
            self.tokens = Dot._tokens
            self.literals = Dot._literals
            self.t_ignore = self.whitespace

        def t_regulars(self, t):
            r"[-]?[\w.]+"
            v = t.value.lower()
            if v in self.reserved:
                t.type = v
                return t
            # check numeric string
            if v[0].isdigit() or v[0] in ["-", "."]:
                try:
                    float(v)
                except ValueError:
                    print("invalid numeral token: %s" % v)
                    raise SyntaxError
            elif "." in v:  # forbidden in non-numeric
                raise SyntaxError
            return t

        def t_comment_online(self, t):
            r"(//(.*)\n)|\\\n"
            pass

        def t_comment_macro(self, t):
            r"(\#(.*)\n)"
            pass

        def t_comment_multline(self, t):
            r"(/\*)"
            start = t.lexer.lexpos
            t.lexer.lexpos = t.lexer.lexdata.index("*/", start) + 2

        def t_string(self, t):
            r'"'
            start = t.lexer.lexpos - 1
            i = t.lexer.lexdata.index('"', start + 1)
            while t.lexer.lexdata[i - 1] == "\\":
                i = t.lexer.lexdata.index('"', i + 1)
            t.value = t.lexer.lexdata[start : i + 1]
            t.lexer.lexpos = i + 1
            return t

        def t_html(self, t):
            r"<"
            start = t.lexer.lexpos - 1
            level = 1
            i = start + 1
            while level > 0:
                c = t.lexer.lexdata[i]
                if c == "<":
                    level += 1
                if c == ">":
                    level -= 1
                i += 1
            t.value = t.lexer.lexdata[start:i]
            t.lexer.lexpos = i
            return t

        def t_ANY_error(self, t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        def build(self, **kargs):
            if _has_ply:
                self._lexer = lex.lex(module=self, **kargs)

        def test(self, data):
            self._lexer.input(data)
            while 1:
                tok = self._lexer.token()
                if not tok:
                    break
                print(tok)

    # Classes for the AST returned by Parser:
    class graph(object):
        def __init__(self, name, data, strict=None, direct=None):
            self.name = name
            self.strict = strict
            self.direct = direct
            self.nodes = {}
            self.edges = []
            self.subgraphs = []
            self.attr = {}
            eattr = {}
            nattr = {}
            for x in data:  # data is a statements (list of stmt)
                # x is a stmt, ie one of:
                # a graph object (subgraph)
                # a attr object (graph/node/edge attributes)
                # a dict object (ID=ID)
                # a node object
                # a list of edges
                if isinstance(x, Dot.graph):
                    self.subgraphs.append(x)
                elif isinstance(x, Dot.attr):
                    if x.type == "graph":
                        self.attr.update(x.D)
                    elif x.type == "node":
                        nattr.update(x.D)
                    elif x.type == "edge":
                        eattr.update(x.D)
                    else:
                        raise TypeError("invalid attribute type")
                elif isinstance(x, dict):
                    self.attr.update(x)
                elif isinstance(x, Dot.node):
                    x.attr.update(nattr)
                    self.nodes[x.name] = x
                else:
                    for e in x:
                        e.attr.update(eattr)
                        self.edges.append(e)
                        for n in [e.n1, e.n2]:
                            if isinstance(n, Dot.graph):
                                continue
                            if n.name not in self.nodes:
                                n.attr.update(nattr)
                                self.nodes[n.name] = n

        def __repr__(self):
            u = "<%s instance at %x, name: %s, %d nodes>" % (
                self.__class__,
                id(self),
                self.name,
                len(self.nodes),
            )
            return u

    class attr(object):
        def __init__(self, type, D):
            self.type = type
            self.D = D

    class edge(object):
        def __init__(self, n1, n2):
            self.n1 = n1
            self.n2 = n2
            self.attr = {}

    class node(object):
        def __init__(self, name, port=None):
            self.name = name
            self.port = port
            self.attr = {}

    class Parser(object):
        def __init__(self):
            self.tokens = Dot._tokens

        def __makelist(self, p):
            N = len(p)
            if N > 2:
                L = p[1]
                L.append(p[N - 1])
            else:
                L = []
                if N > 1:
                    L.append(p[N - 1])
            p[0] = L

        def p_Data(self, p):
            """Data : Data Graph
                    | Graph"""
            self.__makelist(p)

        def p_Graph_strict(self, p):
            """Graph : strict graph name Block"""
            p[0] = Dot.graph(name=p[3], data=p[4], strict=1, direct=0)
            # print 'Dot.Parser: graph object %s created'%p[0].name

        def p_Graph_graph(self, p):
            """Graph : graph name Block"""
            p[0] = Dot.graph(name=p[2], data=p[3], strict=0, direct=0)

        def p_Graph_strict_digraph(self, p):
            """Graph : strict digraph name Block"""
            p[0] = Dot.graph(name=p[3], data=p[4], strict=1, direct=1)

        def p_Graph_digraph(self, p):
            """Graph : digraph name Block"""
            p[0] = Dot.graph(name=p[2], data=p[3], strict=0, direct=1)

        def p_ID(self, p):
            """ID : regulars
                  | string
                  | html """
            p[0] = p[1]

        def p_name(self, p):
            """name : ID
                    | """
            if len(p) == 1:
                p[0] = ""
            else:
                p[0] = p[1]

        def p_Block(self, p):
            """Block : '{' statements '}' """
            p[0] = p[2]

        def p_statements(self, p):
            """statements : statements stmt
                          | stmt
                          | """
            self.__makelist(p)

        def p_stmt(self, p):
            """stmt : stmt ';' """
            p[0] = p[1]

        def p_comment(self, p):
            """stmt : comment"""
            pass  # comment tokens are not outputed by lexer anyway

        def p_stmt_sub(self, p):
            """stmt : sub"""
            p[0] = p[1]

        def p_subgraph(self, p):
            """sub : subgraph name Block
                   | Block """
            N = len(p)
            if N > 2:
                ID = p[2]
            else:
                ID = ""
            p[0] = Dot.graph(name=ID, data=p[N - 1], strict=0, direct=0)

        def p_stmt_assign(self, p):
            """stmt : affect """
            p[0] = p[1]

        def p_affect(self, p):
            """affect : ID '=' ID """
            p[0] = dict([(p[1], p[3])])

        def p_stmt_lists(self, p):
            """stmt : graph attrs
                    | node  attrs
                    | edge  attrs """
            p[0] = Dot.attr(p[1], p[2])

        def p_attrs(self, p):
            """attrs : attrs attrl
                     | attrl """
            if len(p) == 3:
                p[1].update(p[2])
            p[0] = p[1]

        def p_attrl(self, p):
            """attrl : '[' alist ']' """
            L = {}
            for a in p[2]:
                if isinstance(a, dict):
                    L.update(a)
                else:
                    L[a] = "true"
            p[0] = L

        def p_alist_comma(self, p):
            """alist : alist ',' alist """
            p[1].extend(p[3])
            p[0] = p[1]

        def p_alist_affect(self, p):
            """alist : alist affect
                     | alist ID
                     | affect
                     | ID
                     | """
            self.__makelist(p)

        def p_stmt_E_attrs(self, p):
            """stmt : E attrs """
            for e in p[1]:
                e.attr = p[2]
            p[0] = p[1]

        def p_stmt_N_attrs(self, p):
            """stmt : N attrs """
            p[1].attr = p[2]
            p[0] = p[1]

        def p_stmt_EN(self, p):
            """stmt : E
                    | N """
            p[0] = p[1]

        def p_E(self, p):
            """E : E   link
                 | elt link """
            try:
                L = p[1]
                L.append(Dot.edge(L[-1].n2, p[2]))
            except Exception:
                L = []
                L.append(Dot.edge(p[1], p[2]))
            p[0] = L

        def p_elt(self, p):
            """elt : N
                   | sub """
            p[0] = p[1]

        def p_link(self, p):
            """link : '-' '>' elt
                    | '-' '-' elt """
            p[0] = p[3]

        def p_N_port(self, p):
            """N : ID port """
            p[0] = Dot.node(p[1], port=p[2])

        def p_N(self, p):
            """N : ID """
            p[0] = Dot.node(p[1])

        def p_port(self, p):
            """port : ':' ID """
            p[0] = p[2]

        def p_port2(self, p):
            """port : port port"""
            assert p[2] in ["n", "ne", "e", "se", "s", "sw", "w", "nw", "c", "_"]
            p[0] = "%s:%s" % (p[1], p[2])

        def p_error(self, p):
            print("Syntax Error: %s" % (p,))
            self._parser.restart()

        def build(self, **kargs):
            opt = dict(debug=0, write_tables=0)
            opt.update(**kargs)
            if _has_ply:
                self._parser = yacc.yacc(module=self, **opt)

    def __init__(self, **kargs):
        self.lexer = Dot.Lexer()
        self.parser = Dot.Parser()
        if not _has_ply:
            print("warning: Dot parser not supported (install python-ply)")

    def parse(self, data):
        try:
            self.parser._parser.restart()
        except AttributeError:
            self.lexer.build(reflags=lex.re.UNICODE)
            self.parser.build()
        except Exception:
            print("unexpected error")
            return None
        try:
            s = data.decode("utf-8")
        except UnicodeDecodeError:
            s = data
        L = self.parser._parser.parse(s, lexer=self.lexer._lexer)
        return L

    def read(self, filename):
        f = open(
            filename, "rb"
        )  # As it'll try to decode later on with utf-8, read it binary at this point.
        return self.parse(f.read())
