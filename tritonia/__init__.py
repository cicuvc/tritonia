from typing import Any, Callable, Generator, Iterable, List, Tuple, Union
import triton
import triton.language as tl
import torch
import ast
import inspect
import builtins

import numpy as np





PrimaryStatement = Union[ast.Call, ast.Assign]

class EvaluatedValue:
    def __init__(self, value: Union[type["EvaluatedValue"], Any]):
        assert not isinstance(value, ast.AST)

        if isinstance(value, EvaluatedValue):
            value = value.value
        self.value = value
        self.ast_optional = InterpreterHelpers.remove_const_to_ast(value, ret_none=True)


class IdentifierRefValue:
    def __init__(self, ast_node: Union[ast.Name, ast.expr]):
        super().__init__()
        self.ast_node = ast_node

class PseudoTensorValue(IdentifierRefValue):
    def __init__(self, ast_node: Union[ast.Name, ast.expr]):
        super().__init__(ast_node)

    def __add__(self, rhs: Union[type["PseudoTensorValue"], ast.stmt, int, float]):
        if type(rhs) in (int, float):
            rhs = EvaluatedValue(rhs)
        return PseudoTensorValue(ast.BinOp(self, ast.Add(), rhs))
    def __radd__(self, rhs: Union[type["PseudoTensorValue"], ast.stmt, int, float]):
        if type(rhs) in (int, float):
            rhs = EvaluatedValue(rhs)
        return PseudoTensorValue(ast.BinOp(self, ast.Add(), rhs))

    def __mul__(self, rhs: Union[type["PseudoTensorValue"], int, float]):
        if type(rhs) in (int, float):
            rhs = EvaluatedValue(rhs)
        return PseudoTensorValue(ast.BinOp(self, ast.Mult(), rhs))
    def __rmul__(self, lhs: Union[type["PseudoTensorValue"], int, float]):
        if type(lhs) in (int, float):
            lhs = EvaluatedValue(lhs)
        return PseudoTensorValue(ast.BinOp(lhs, ast.Mult(), self))
    

    def __truediv__(self, rhs: Union[type["PseudoTensorValue"], int, float]):
        if type(rhs) in (int, float, np.int64):
            rhs = EvaluatedValue(rhs)
        return PseudoTensorValue(ast.BinOp(self, ast.Div(), rhs))
    def __rtruediv__(self, lhs: Union[type["PseudoTensorValue"], int, float]):
        if type(lhs) in (int, float):
            lhs = EvaluatedValue(lhs)
        
        return PseudoTensorValue(ast.BinOp(lhs, ast.Div(), self))
        
    def __str__(self):
        return self.ast_node.__str__()
    
    def __repr__(self):
        return self.ast_node.id

ExtraScope = dict[str, EvaluatedValue]

class InterpreterHelpers:
    bin_op_dispatchers: dict[type, Callable[[EvaluatedValue, EvaluatedValue], EvaluatedValue]] = {
        ast.Add: lambda lhs, rhs:   EvaluatedValue(lhs.value + rhs.value),
        ast.Sub: lambda lhs, rhs:   EvaluatedValue(lhs.value - rhs.value),
        ast.Mult: lambda lhs, rhs:  EvaluatedValue(lhs.value * rhs.value),
        ast.Div: lambda lhs, rhs:   EvaluatedValue(lhs.value / rhs.value),
        ast.MatMult: lambda lhs, rhs: EvaluatedValue(lhs.value @ rhs.value),
        ast.Mod: lambda lhs, rhs:   EvaluatedValue(lhs.value % rhs.value),
        ast.Pow: lambda lhs, rhs:   EvaluatedValue(lhs.value ** rhs.value),
        ast.And: lambda lhs, rhs:   EvaluatedValue(lhs.value and rhs.value),
        ast.Or: lambda lhs, rhs:    EvaluatedValue(lhs.value or rhs.value),
        ast.BitAnd: lambda lhs, rhs: EvaluatedValue(lhs.value & rhs.value),
        ast.BitOr: lambda lhs, rhs: EvaluatedValue(lhs.value | rhs.value),
        ast.BitXor: lambda lhs, rhs: EvaluatedValue(lhs.value ^ rhs.value),
    }
    multi_op_dispatchers: dict[type, Callable[[Iterable[EvaluatedValue]], EvaluatedValue]] = {
        ast.And: lambda v: EvaluatedValue(all([i.value for i in v])),
        ast.Or: lambda v: EvaluatedValue(any([i.value for i in v])),
    }
    compare_op_dispatchers: dict[type, Callable[[EvaluatedValue, Iterable[EvaluatedValue]], bool]] = {
        ast.Eq: lambda lhs, v: EvaluatedValue(lhs == next(iter(v))),
        ast.NotEq: lambda lhs, v: EvaluatedValue(lhs != next(iter(v))),
        ast.In: lambda lhs, v: EvaluatedValue(lhs in iter(v))
    }
    @staticmethod
    def binary_op_dispatch(lhs: EvaluatedValue, rhs: EvaluatedValue, op: ast.operator) -> EvaluatedValue:
        return InterpreterHelpers.bin_op_dispatchers[type(op)](lhs, rhs)
    
    @staticmethod
    def multi_op_dispatch(values: Iterable[EvaluatedValue], op: ast.operator) -> EvaluatedValue:
        return InterpreterHelpers.multi_op_dispatchers[type(op)](values)
    
    @staticmethod
    def compare_op_dispatch(lhs: EvaluatedValue,values: Iterable[EvaluatedValue], ops: Iterable[ast.operator]) -> EvaluatedValue:
        res = True
        for i in ops:
            res &= InterpreterHelpers.compare_op_dispatchers[type(i)](lhs, values)
        return EvaluatedValue(res)
    
    @staticmethod
    def is_const(ast_node: Union[EvaluatedValue, PseudoTensorValue, Any]):
        return isinstance(ast_node, (EvaluatedValue, PseudoTensorValue))
    
    @staticmethod
    def is_evalueted_const(ast_node: Union[EvaluatedValue, Any]):
        return isinstance(ast_node, (EvaluatedValue))
    
    @staticmethod
    def remove_const(x: Union[EvaluatedValue, PseudoTensorValue], allow_ast: bool = True, throw: bool = False):
        if isinstance(x, EvaluatedValue):
            return x.value
        if isinstance(x, PseudoTensorValue) and allow_ast:
            return x.ast_node
        if throw:
            raise NotImplementedError()
        return x
    
    @staticmethod
    def remove_const_to_ast(x: Union[EvaluatedValue, PseudoTensorValue], allow_ast: bool = True, throw: bool = False, ret_none: bool = False):
        if isinstance(x, (int, float, str, np.int64)):
                return ast.Constant(x)
        if isinstance(x, EvaluatedValue):
            value = x.value
            if isinstance(value, (int, float, str, np.int64)):
                return ast.Constant(value)
            if isinstance(value, PseudoTensorValue):
                return value.ast_node
            raise NotImplementedError()
        if isinstance(x, PseudoTensorValue) and allow_ast:
            return x.ast_node
        if throw:
            raise NotImplementedError()
        if ret_none:
            return None
        return x
    
    @staticmethod
    def expand_args(args: list):
        return [InterpreterHelpers.remove_const(i, allow_ast=False, throw=False) for i in args]
    @staticmethod
    def expand_kwargs(args: dict):
        return {k: InterpreterHelpers.remove_const(v, allow_ast=False, throw=False) for (k,v) in args.items()}


class SemiInterpretedAstTransformer:
    def __init__(self, ast_func: ast.FunctionDef, determined_args: dict[str, Any]):
        self.ast_func = ast_func
        self.cached_vars: dict[ast.stmt, str] = dict()
        self.dependency_set: dict[PrimaryStatement, List[ast.stmt]] = dict()
        self.local_vars: dict[str, Union[PseudoTensorValue, EvaluatedValue]] = dict()
        self.name_scope = dict(globals())
        self.name_scope.update({
            k: getattr(builtins, k) for k in dir(builtins)
        })
        self.determined_args = determined_args

        self.no_eval_funcs = {tl.arange, tl.load, tl.store}

        self.expr_eval_funcs: dict[ast.stmt, Callable[[ast.stmt, ExtraScope, PrimaryStatement], Union[PseudoTensorValue, EvaluatedValue]]] = {
            ast.Call: self.eval_call,
            ast.BinOp: self.eval_binop,
            ast.Name: self.eval_name,
            ast.Attribute: self.eval_attribute,
            ast.Constant: lambda x, e, p: EvaluatedValue(x.value),
            ast.BoolOp: self.eval_compare_op,
            ast.Compare: self.eval_compare_op,
            ast.ListComp: self.eval_listcomp,
            ast.Tuple: self.eval_tuple,
            ast.Subscript: self.eval_subscr
        }
        self.func_name = ast_func.name

    def allocate_cache(self, x: ast.stmt, primary: PrimaryStatement) -> ast.Name:
        identifier = f"local_{len(self.cached_vars)}"
        self.cached_vars[x] = identifier
        if not (primary in self.dependency_set):
            self.dependency_set[primary] = []
        self.dependency_set[primary].append(x)

        return ast.Name(identifier, ctx = ast.Load())

    def eval_any(self, ast_node: ast.stmt, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[PseudoTensorValue, EvaluatedValue]:
        return self.expr_eval_funcs[type(ast_node)](ast_node, extra_scope, primary)
    def eval_tuple(self, ast_tuple: ast.Tuple, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[PseudoTensorValue, EvaluatedValue]:
        values = [self.eval_any(i, extra_scope, primary) for i in ast_tuple.elts]
        if all([InterpreterHelpers.is_evalueted_const(i) for i in values]):
            return EvaluatedValue(tuple([InterpreterHelpers.remove_const(i) for i in values]))
        raise NotImplementedError()
    
    def eval_subscr(self, ast_subscr: ast.Subscript, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[PseudoTensorValue, EvaluatedValue]:
        base_value = self.eval_any(ast_subscr.value, extra_scope, primary)
        slices = self.eval_any(ast_subscr.slice, extra_scope, primary)
        if hasattr(slices, '__iter__'):
            result = InterpreterHelpers.remove_const(base_value)[*InterpreterHelpers.remove_const(slices)]
        else:
            result = InterpreterHelpers.remove_const(base_value)[InterpreterHelpers.remove_const(slices)]

        return EvaluatedValue(result)
    
    def traverse_node(self, ast_node: ast.AST):
        pass
    
    def process_expr(self, ast_expr: ast.Expr) -> List[ast.stmt]:
        value = self.eval_any(ast_expr.value, dict(), ast_expr)

        result = []
        if ast_expr in self.dependency_set:
            result.extend([
                ast.Assign([ast.Name(self.cached_vars[v], ast.Store())], v, lineno=0) for v in self.dependency_set[ast_expr]
                ])

        return result
    
    def process_func(self, ast_node: ast.FunctionDef):
        new_body = []
        for i in self.ast_func.body:
            if type(i) == ast.Call:
                self.process_call(i)
            elif type(i) == ast.Assign:
                new_body.extend(self.process_assign(i, dict()))
            elif type(i) == ast.Expr:
                new_body.extend(self.process_expr(i))
            elif type(i) == ast.AugAssign:
                new_body.extend(self.process_aug_assign(i, dict()))
            else:
                raise NotImplementedError()
            
        jit_decorator = ast.Attribute(ast.Name('triton', ast.Load()), 'jit', ast.Load())
        return ast.FunctionDef(ast_node.name, ast_node.args, new_body, [jit_decorator], lineno=0)
            
    def process_call(self, ast_call: ast.Call) -> None:
        self.eval_call(ast_call, dict(), ast_call)
        
    def process_assign(self, ast_assign: ast.Assign, extra_scope: ExtraScope) -> list[ast.stmt]:
        store_targets = [self.eval_store_name(i) for i in ast_assign.targets]
        if len(store_targets) > 1:
            raise NotImplementedError()
        
        rhs = self.eval_any(ast_assign.value, extra_scope, ast_assign)
        store_targets[0](rhs)

        result = []
        if ast_assign in self.dependency_set:
            result.extend([
                ast.Assign([ast.Name(self.cached_vars[v], ast.Store())], v, lineno=0) for v in self.dependency_set[ast_assign]
                ])
        
        rhs = InterpreterHelpers.remove_const(rhs)
        if isinstance(rhs, ast.AST):
            result.append(ast.Assign(ast_assign.targets, rhs, lineno=0))

        return result
    
    def process_aug_assign(self, ast_assign: ast.AugAssign, extra_scope: ExtraScope) -> list[ast.stmt]:
        store_targets = self.eval_store_name(ast_assign.target)
        lhs = self.eval_any(ast_assign.target, extra_scope, ast_assign)
        rhs = self.eval_any(ast_assign.value, extra_scope, ast_assign)

        if InterpreterHelpers.is_evalueted_const(lhs) and InterpreterHelpers.is_evalueted_const(rhs):
            new_val = InterpreterHelpers.binary_op_dispatch(lhs, rhs, ast_assign.op)
            store_targets(new_val)

        result = []
        if ast_assign in self.dependency_set:
            result.extend([
                ast.Assign([ast.Name(self.cached_vars[v], ast.Store())], v, lineno=0) for v in self.dependency_set[ast_assign]
                ])
        
        rhs = InterpreterHelpers.remove_const(rhs)
        if isinstance(rhs, ast.AST):
            result.append(ast.AugAssign(ast_assign.targets, ast_assign.op, rhs, lineno=0))

        return result

    def eval_call(self, ast_call: ast.Call, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[PseudoTensorValue, EvaluatedValue]:
        callee = self.eval_any(ast_call.func, extra_scope, primary)
        args = [self.eval_any(i, extra_scope, primary) for i in ast_call.args]
        kwargs = {i.arg: self.eval_any(i.value, extra_scope) for i in ast_call.keywords}
        
        if not InterpreterHelpers.is_const(callee):
            raise NotImplementedError()
        
        if (callee.value in self.no_eval_funcs):
            kws = [ast.keyword(k, v) for (k,v) in kwargs.items()]
            new_call = ast.Call(ast_call.func, args, kws)
            return PseudoTensorValue(self.allocate_cache(new_call, primary))

        ret = callee.value(*InterpreterHelpers.expand_args(args), **InterpreterHelpers.expand_kwargs(kwargs))
    
        if isinstance(ret, PseudoTensorValue):
            return PseudoTensorValue(self.allocate_cache(ret.ast_node, primary))
        return EvaluatedValue(ret) 
    
    def eval_listcomp(self, ast_comp: ast.ListComp, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[PseudoTensorValue, EvaluatedValue]:
        generators = [self.eval_comprehension(i, extra_scope, primary) for i in ast_comp.generators]
        iterators = []
        values = []
        result = []
        scope = dict(extra_scope)

        iterators.append(iter(generators[0][1].value))
        values.append(None)

        while len(values) > 0:
            values[-1] = next(iterators[-1], None)
            if values[-1] is None:
                del values[-1]
                del iterators[-1]
                continue

            for i in range(len(values), len(generators)):
                iterator = iter(generators[i][1].value)
                iterators.append(iterator)
                nxt_val = next(iterator, None)
                values.append(nxt_val)

            if any([i is None for i in values]):
                break
            scope.update({
                generators[i][0]: values[i] for i in range(len(generators))
            })

            if all([i[2](scope) for i in generators]):
                item = self.eval_any(ast_comp.elt, scope, primary)
                result.append(item)

        return EvaluatedValue(result)

    def eval_comprehension(self, ast_comp: ast.comprehension, extra_scope: ExtraScope, primary: PrimaryStatement) -> Tuple[str, Iterable[EvaluatedValue], Callable[[ExtraScope], bool]]:
        target: str = ast_comp.target.id
        iterable = self.eval_any(ast_comp.iter, extra_scope, primary)
        def condition(scope: ExtraScope):
            return all([self.eval_any(i, extra_scope, primary) for i in ast_comp.ifs])
        
        return (target, iterable, condition)

    def eval_bool_op(self, ast_boolop: ast.BoolOp, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[ast.stmt, EvaluatedValue]:
        values = [self.eval_any(i, extra_scope, primary) for i in ast_boolop.values]
        if all([InterpreterHelpers.is_evalueted_const(i) for i in values]):
            return InterpreterHelpers.multi_op_dispatch(values, ast_boolop.op)
        return PseudoTensorValue(ast.BoolOp(ast_boolop.op, values))
    
    def eval_compare_op(self, ast_comp: ast.Compare,extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[ast.stmt, EvaluatedValue]:
        lhs = self.eval_any(ast_comp.left, extra_scope, primary)
        values = [self.eval_any(i, extra_scope, primary) for i in ast_comp.comparators]
        if InterpreterHelpers.is_evalueted_const(lhs) and all([InterpreterHelpers.is_evalueted_const(i) for i in values]):
            return InterpreterHelpers.compare_op_dispatch(lhs, values, ast_comp.ops)
        return PseudoTensorValue(ast.Compare(lhs, values, ast_comp.ops))

    def eval_binop(self, ast_binop: ast.BinOp, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[PseudoTensorValue, EvaluatedValue]:
        lhs = self.eval_any(ast_binop.left, extra_scope, primary)
        rhs = self.eval_any(ast_binop.right, extra_scope, primary)
        if InterpreterHelpers.is_evalueted_const(lhs) and InterpreterHelpers.is_evalueted_const(rhs):
            return InterpreterHelpers.binary_op_dispatch(lhs, rhs, ast_binop.op)
        return PseudoTensorValue(ast.BinOp(InterpreterHelpers.remove_const_to_ast(lhs), ast_binop.op, InterpreterHelpers.remove_const_to_ast(rhs)))

    def eval_name(self, ast_name: ast.Name, extra_scope: ExtraScope, primary: PrimaryStatement) -> EvaluatedValue:
        if ast_name.id in extra_scope:
            return EvaluatedValue(extra_scope[ast_name.id])
        if ast_name.id in self.local_vars:
            local = (self.local_vars[ast_name.id])
            assert isinstance(local, (EvaluatedValue, PseudoTensorValue))
            return local
        if ast_name.id in self.name_scope:
            return EvaluatedValue(self.name_scope[ast_name.id])
        return ast_name

    def eval_attribute(self, ast_attr: ast.Attribute, extra_scope: ExtraScope, primary: PrimaryStatement) -> Union[ast.stmt, EvaluatedValue]:
        attr_parent = self.eval_any(ast_attr.value, extra_scope, primary)
        if not InterpreterHelpers.is_const(attr_parent):
            return ast_attr
        return EvaluatedValue(getattr(attr_parent.value, ast_attr.attr))

    def eval_store_name(self, ast_name: ast.Name) -> Callable[[Any], None]:
        def store(x):
            self.local_vars[ast_name.id] = x
        return store
    
class CodeWriter(ast._Unparser):
    def __init__(self, *args):
        super().__init__(*args)

    def traverse(self, node):
        if type(node) == PseudoTensorValue:
            self.traverse(node.ast_node)
        elif isinstance(node, EvaluatedValue):
            assert node.ast_optional is not None

            self.traverse(node.ast_optional)
        else:
            super().traverse(node)
    def set_precedence(self, precedence, *nodes):
        for node in nodes:
            if isinstance(node, PseudoTensorValue):
                self._precedences[node.ast_node] = precedence
            elif isinstance(node, EvaluatedValue):
                self._precedences[node.ast_optional] = precedence
            else:
                super().set_precedence(precedence, *nodes)


def ijit(fn):
    astx = ast.parse(inspect.getsource(fn))
    print(ast.dump(astx, indent=4))
    tf = SemiInterpretedAstTransformer(astx.body[0], {})
    new_ast = tf.process_func(tf.ast_func)
    unparser = CodeWriter()
    source: str = unparser.visit(new_ast)
    parse_globals = dict(globals())
    parse_locals = dict()
    parse_globals.update({
        k: getattr(builtins, k) for k in dir(builtins)
    })
    origin_findsource = inspect.findsource
    def new_findsource(obj) -> Tuple[List[str], int]:
        if isinstance(obj, type(new_findsource)):
            return ([i + '\n' for i in source.split('\n')], 1)
        else:
            return origin_findsource(obj)
    inspect.findsource = new_findsource
    exec(source, parse_globals, parse_locals)
    inspect.findsource = origin_findsource
    return parse_locals[tf.func_name]
        
@ijit
def add_kernel(x_ptr, y_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x_values = np.array([tl.load(x_ptr + offsets * 4 + i) for i in range(4)]).reshape((2,2))
    y_values = np.array([tl.load(y_ptr + offsets * 4 + i) for i in range(4)]).reshape((2,2))
    x_reduced = (y_values @ x_values).flatten()

    [tl.store(x_ptr + offsets * 4 + i, x_reduced[i]) for i in range(4)]
