import ast
import builtins
from typing import List, Dict, Any, Set, Optional


class PythonProgramAnalyzer:
    def __init__(self, tool_names: Optional[List[str]] = None):
        # 工具函数名集合：这些名字在调用处不会被当作未定义变量报错
        self.tool_names: Set[str] = set(tool_names or [])
        self.all_functions: List[str] = []  # 代码中通过 def/async def 定义的函数名
        self.tool_sequence: List[str] = []  # 代码中调用顺序里的工具函数名（在 tool_names 中）
        self.errors: List[Dict[str, Any]] = []  # 所有错误（含行列信息）

    def analyze(self, code_str: str) -> Dict[str, Any]:
        # 重置状态
        self.all_functions = []
        self.tool_sequence = []
        self.errors = []
        invalid_variables: Set[str] = set()
        all_vars: Set[str] = set()

        # 1) 语法检查
        try:
            tree = ast.parse(code_str)
            is_valid = True
        except SyntaxError as e:
            self.errors.append({
                "type": "SyntaxError",
                "message": str(e),
                "lineno": getattr(e, "lineno", None),
                "col_offset": getattr(e, "offset", None),
                "text": getattr(e, "text", None),
            })
            return {
                "validity": False,
                "functions": [],
                "tool_sequence": [],
                "variables": {"all": [], "invalid": []},
                "errors": self.errors,
            }

        # 2) 收集定义的函数名
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.all_functions.append(node.name)

        # 2.1) 按调用顺序收集工具函数
        def call_func_name(n: ast.AST) -> Optional[str]:
            # 支持: foo() / mod.foo() -> 取 'foo'
            if isinstance(n, ast.Name):
                return n.id
            if isinstance(n, ast.Attribute):
                return n.attr
            return None

        def preorder(n: ast.AST):
            # 前序遍历以保持出现顺序
            if isinstance(n, ast.Call):
                fn = call_func_name(n.func)
                if fn and fn in self.tool_names:
                    self.tool_sequence.append(fn)
            for ch in ast.iter_child_nodes(n):
                preorder(ch)

        preorder(tree)

        # 3) 变量与作用域分析
        builtins_set = set(dir(builtins))

        class Scope:
            def __init__(self, parent: Optional["Scope"] = None):
                self.parent = parent
                self.defined: Set[str] = set()

            def is_defined(self, name: str) -> bool:
                if name in self.defined:
                    return True
                return self.parent.is_defined(name) if self.parent else False

        global_scope = Scope()

        def define_name(scope: Scope, name: str):
            scope.defined.add(name)
            all_vars.add(name)

        def define_target(scope: Scope, target: ast.AST):
            # 赋值/目标绑定生成名字
            if isinstance(target, ast.Name):
                define_name(scope, target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    define_target(scope, elt)
            # Attribute/Subscript 不产生新绑定

        def handle_comprehension_elt_scope(generators, outer_scope: Scope) -> Scope:
            """
            对于推导式/生成器表达式，按生成子句依次创建内层作用域，
            在每层作用域里绑定 target，访问 if 条件，然后把作用域传给下一层。
            返回“最内层作用域”。
            """
            scope = outer_scope
            for gen in generators:
                # iter 在当前 scope 下访问（可能读取外部名）
                visit(gen.iter, scope)
                # 目标绑定与 if 条件在新的内层作用域
                inner = Scope(scope)
                define_target(inner, gen.target)
                for cond in gen.ifs:
                    visit(cond, inner)
                scope = inner  # 下一轮在更内层进行
            return scope

        def visit(node: ast.AST, scope: Scope):
            # ---- 函数定义 ----
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                define_name(scope, node.name)
                fn_scope = Scope(scope)
                # 参数名绑定
                for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    define_name(fn_scope, a.arg)
                if node.args.vararg:
                    define_name(fn_scope, node.args.vararg.arg)
                if node.args.kwarg:
                    define_name(fn_scope, node.args.kwarg.arg)
                # 函数体
                for b in node.body:
                    visit(b, fn_scope)
                return

            # ---- 类定义（简化为新作用域）----
            if isinstance(node, ast.ClassDef):
                define_name(scope, node.name)
                cls_scope = Scope(scope)
                for b in node.body:
                    visit(b, cls_scope)
                return

            # ---- 变量赋值 ----
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    define_target(scope, t)
                visit(node.value, scope)
                return

            if isinstance(node, ast.AnnAssign):
                if node.target:
                    define_target(scope, node.target)
                if node.value:
                    visit(node.value, scope)
                return

            if isinstance(node, ast.AugAssign):
                # 读后写：先视作读取（检测未定义），后写入绑定
                if isinstance(node.target, ast.Name):
                    name = node.target.id
                    if (
                            name not in builtins_set
                            and name not in self.tool_names
                            and not scope.is_defined(name)
                    ):
                        invalid_variables.add(name)
                        self.errors.append({
                            "type": "NameError",
                            "message": f"Variable '{name}' used before assignment.",
                            "lineno": node.lineno,
                            "col_offset": node.col_offset,
                        })
                    define_name(scope, name)
                visit(node.value, scope)
                return

            # ---- for 循环 ----
            if isinstance(node, ast.For):
                visit(node.iter, scope)
                define_target(scope, node.target)
                for b in node.body:
                    visit(b, scope)
                for o in node.orelse:
                    visit(o, scope)
                return

            # ---- with 语句 ----
            if isinstance(node, ast.With):
                for item in node.items:
                    visit(item.context_expr, scope)
                    if item.optional_vars:
                        define_target(scope, item.optional_vars)
                for b in node.body:
                    visit(b, scope)
                return

            # ---- try/except 中的异常变量 ----
            if isinstance(node, ast.ExceptHandler):
                if node.name:
                    define_name(scope, node.name)
                for b in node.body:
                    visit(b, scope)
                return

            # ---- import 绑定 ----
            if isinstance(node, ast.Import):
                for alias in node.names:
                    define_name(scope, alias.asname or alias.name.split(".")[0])
                return

            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    define_name(scope, alias.asname or alias.name)
                return

            # ---- lambda 作用域 ----
            if isinstance(node, ast.Lambda):
                lam_scope = Scope(scope)
                for a in node.args.args:
                    define_name(lam_scope, a.arg)
                # 仅 body 是表达式
                visit(node.body, lam_scope)
                return

            # ---- 推导式 / 生成器表达式 ----
            if isinstance(node, ast.ListComp):
                inner = handle_comprehension_elt_scope(node.generators, scope)
                visit(node.elt, inner)
                return

            if isinstance(node, ast.SetComp):
                inner = handle_comprehension_elt_scope(node.generators, scope)
                visit(node.elt, inner)
                return

            if isinstance(node, ast.DictComp):
                inner = handle_comprehension_elt_scope(node.generators, scope)
                visit(node.key, inner)
                visit(node.value, inner)
                return

            if isinstance(node, ast.GeneratorExp):
                inner = handle_comprehension_elt_scope(node.generators, scope)
                visit(node.elt, inner)
                return

            # ---- 名字读取（Load）检测未定义 ----
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if (
                        name not in builtins_set
                        and name not in self.tool_names  # 工具名不当作未定义
                        and not scope.is_defined(name)
                ):
                    invalid_variables.add(name)
                    self.errors.append({
                        "type": "NameError",
                        "message": f"Variable '{name}' used before definition.",
                        "lineno": node.lineno,
                        "col_offset": node.col_offset,
                    })
                return

            # 其他节点：递归遍历
            for ch in ast.iter_child_nodes(node):
                visit(ch, scope)

        # 启动遍历
        visit(tree, global_scope)

        return {
            "validity": is_valid,
            "functions": self.all_functions,
            "tool_sequence": self.tool_sequence,
            "variables": {
                "all": sorted(all_vars),
                "invalid": sorted(invalid_variables),
            },
            "errors": self.errors,
        }


# -------------------------
# 使用示例
if __name__ == "__main__":
    code = """
# 顶层脚本示例（无 def）
time_phrase = extract_time_from_query(query="How many days did I skip commuting in the mornings?")
time_window = resolve_time_window(time_text=time_phrase if time_phrase else "this week")
start_time = time_window['start']; end_time = time_window['end']
morning_contexts = filter_contexts_by_time(start=start_time, end=end_time)

conditions = {'action': 'commute'}
negative_conditions_existence = not check_existence(contexts=morning_contexts, conditions=conditions)

if negative_conditions_existence:
    skipped_days_count = 0
    split_windows = split_time_window(start=start_time, end=end_time, freq_unit='day', freq_value=1)
    for window in split_windows:
        day_contexts = filter_contexts_by_time(start=window['start'], end=window['end'])
        exists = check_existence(contexts=day_contexts, conditions=conditions)
        if not exists:
            skipped_days_count += 1
else:
    skipped_days_count = 0

# 推导式/生成器表达式与 lambda
last_time = max(context['end'] for context in morning_contexts if 'end' in context)
start_exercise_datetime = min(exercise_durations, key=lambda x: x['start'])['start']
"""

    tools = [
        "extract_time_from_query", "resolve_time_window", "filter_contexts_by_time",
        "check_existence", "split_time_window"
    ]
    analyzer = PythonProgramAnalyzer(tool_names=tools)
    from pprint import pprint

    pprint(analyzer.analyze(code))
