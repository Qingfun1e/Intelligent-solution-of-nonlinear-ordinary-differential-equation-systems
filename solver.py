import sympy as sp
import numpy as np
import re
from scipy.integrate import solve_ivp
class ODESolver:
    def __init__(self, f_x_text, coefficients, n):
        self.f_x_text = f_x_text
        self.coefficients = coefficients
        self.n = n
        self.x, self.y = sp.symbols('x y')
        self.f_expr = self.parse_function_from_text(f_x_text)

        if self.f_expr is None:
            raise ValueError("无法解析函数表达式")

        if len(coefficients) != n + 1:
            raise ValueError(f"常系数列表的长度应为 {n + 1}，但实际为 {len(coefficients)}")

        # 将符号表达式转换为数值函数
        self.f_func = sp.lambdify((self.x, self.y), self.f_expr, 'numpy')

    def parse_function_from_text(self, text):
        """
        从文本中解析出函数 f(x, y)
        :param text: 输入的文本，包含函数表达式
        :return: 符号表达式 f(x, y)
        """
        # 定义变量
        x, y = sp.symbols('x y')

        # 定义基本函数
        basic_functions = {
            'exp': sp.exp,
            'log': sp.log,
            'ln': sp.log,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'csc': sp.csc,
            'sec': sp.sec,
            'cot': sp.cot,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'acsc': sp.acsc,
            'asec': sp.asec,
            'acot': sp.acot,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'csch': sp.csch,
            'sech': sp.sech,
            'coth': sp.coth,
            'asinh': sp.asinh,
            'acosh': sp.acosh,
            'atanh': sp.atanh,
            'acsch': sp.acsch,
            'asech': sp.asech,
            'acoth': sp.acoth,
            'Derivative': sp.Derivative
        }

        # 预处理文本
        text = text.replace('^', '**')  # 将 ^ 替换为 **

        # 使用正则表达式匹配函数表达式
        match = re.search(r'f\s*\(\s*x\s*,\s*y\s*\)\s*=\s*(.*)', text)
        if match:
            function_str = match.group(1)
            try:
                # 将字符串转换为符号表达式
                function_expr = sp.sympify(function_str, locals=basic_functions)
                return function_expr
            except sp.SympifyError:
                print("无法解析函数表达式")
                return None
        else:
            print("未找到函数表达式")
            return None

    def ode_system(self, t, y):
        eplision = 1e-5
        # y 是一个包含 n 个元素的向量，y = [y, y', y'', ...]
        dydt = [y[i] for i in range(1,self.n)]
        # 构造微分方程的右侧部分
        right_side = (self.f_func(t, y[0]) - sum(self.coefficients[i] * y[i] for i in range(self.n))) / \
                     (self.coefficients[-1] + eplision)
        dydt.append(right_side)
        return dydt

    def calculate_highest_order_initial(self, initial_conditions):
        """
        计算最高阶导数的初始值
        :param initial_conditions: 初始条件列表，长度应为阶数 n-1
        :return: 最高阶导数的初始值
        """
        eplision = 1e-5
        t0 = 0  # 初始时间
        highest_order_initial = (self.f_func(t0, initial_conditions[0]) - sum(self.coefficients[i]*initial_conditions[i] for i in range(self.n))) / \
                                (self.coefficients[-1] + eplision)
        return highest_order_initial

    def solve(self, initial_conditions ,t_eval,t_span=(0, 10)):
        """
        使用 RK45 方法求解微分方程
        :param initial_conditions: 初始条件列表，长度应为阶数 n-1
        :param t_span: 时间区间 (t_start, t_end)
        :return: 求解结果
        """
        if len(initial_conditions) != self.n:
            raise ValueError(f"初始条件的数量应为 {self.n} 个，但实际为 {len(initial_conditions)} 个")

        # 使用 RK45 方法求解微分方程
        sol = solve_ivp(self.ode_system, t_span, initial_conditions, method='RK45',t_eval=np.linspace(*t_span, t_eval))
        return sol

if __name__ == "__main__":
    # 示例使用
    text = "f(x, y) = x"
    coefficients = [1, -3, 2]  # 常系数 [A, B, C]
    n = 2  # 阶数
    initial_conditions = [0, 1]  # 初始条件 [y(0), y'(0)]
    t_span = (0, 100)  # 时间区间，从时间 0 到时间 100

    solver = ODESolver(text, coefficients, n)
    solution = solver.solve(initial_conditions, t_span)
    import matplotlib.pyplot as plt

    print("f(x):", solver.f_expr)
    print("t:", solution.t)
    print("y:", solution.y)
    # 绘制结果
    plt.plot(solution.t, solution.y[0], label='y(t)')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.title('Solution of the ODE 2y\'\' - 3y\' + y = x')
    plt.show()