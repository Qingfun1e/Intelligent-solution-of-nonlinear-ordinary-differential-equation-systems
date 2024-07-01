import os
import shutil
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QWidget, QDialog, QFormLayout, QLineEdit, QLabel,
                             QMessageBox, QHBoxLayout)
from solver import ODESolver
from train import train_model
from models import create_model
from datahandler import CustomDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import torch


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("智能解常微分方程系统")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()
        button_layout = QVBoxLayout()

        # Create buttons based on the diagram
        self.button_analyze = QPushButton('解析')
        self.button_draw = QPushButton('绘制曲线')
        self.button_instructions = QPushButton('使用说明')
        self.button_generate_data = QPushButton('生成数据集')
        self.button_train_model = QPushButton('训练模型')
        self.button_infer = QPushButton('推理')

        button_layout.addWidget(self.button_analyze)
        button_layout.addWidget(self.button_draw)
        button_layout.addWidget(self.button_generate_data)
        button_layout.addWidget(self.button_train_model)
        button_layout.addWidget(self.button_infer)
        button_layout.addWidget(self.button_instructions)
        # Connect buttons to their respective functions
        self.button_analyze.clicked.connect(self.open_analyze_dialog)
        self.button_draw.clicked.connect(self.plot_solution)
        self.button_instructions.clicked.connect(self.show_instructions)
        self.button_generate_data.clicked.connect(self.generate_data)
        self.button_train_model.clicked.connect(self.open_train_dialog)
        self.button_infer.clicked.connect(self.infer_model)

        button_container = QWidget()
        button_container.setLayout(button_layout)
        main_layout.addWidget(button_container)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.solution = None
        self.model = None
        self.dataset = None
        # Apply stylesheet
        self.apply_stylesheet()

    def apply_stylesheet(self):
        stylesheet = """
            QMainWindow {
                background-color: #e0f7fa; /* 主界面背景色 */
            }
            QPushButton {
                color: white;
                font-size: 16px;
                border: none;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
            }
            QPushButton#button_analyze {
                background-color: #00796b; /* 解析按钮颜色 */
            }
            QPushButton#button_analyze:hover {
                background-color: #004d40;
            }
            QPushButton#button_draw {
                background-color: #0288d1; /* 绘制曲线按钮颜色 */
            }
            QPushButton#button_draw:hover {
                background-color: #01579b;
            }
            QPushButton#button_instructions {
                background-color: #303f9f; /* 使用说明按钮颜色 */
            }
            QPushButton#button_instructions:hover {
                background-color: #1a237e;
            }
            QPushButton#button_generate_data {
                background-color: #512da8; /* 生成数据集按钮颜色 */
            }
            QPushButton#button_generate_data:hover {
                background-color: #311b92;
            }
            QPushButton#button_train_model {
                background-color: #1976d2; /* 训练模型按钮颜色 */
            }
            QPushButton#button_train_model:hover {
                background-color: #0d47a1;
            }
            QPushButton#button_infer {
                background-color: #0097a7; /* 推理按钮颜色 */
            }
            QPushButton#button_infer:hover {
                background-color: #006064;
            }
            QPushButton#solve_button {
                background-color: #0288d1; /* 解析并求解按钮颜色 */
            }
            QPushButton#solve_button:hover {
                background-color: #01579b;
            }
            QPushButton#train_button {
                background-color: #1976d2; /* 训练按钮颜色 */
            }
            QPushButton#train_button:hover {
                background-color: #0d47a1;
            }
            QDialog {
                background-color: #e0f7fa; /* 子窗口背景色 */
            }
            QLineEdit {
                padding: 5px;
                font-size: 14px;
            }
            QLabel {
                font-size: 14px;
            }
        """
        self.setStyleSheet(stylesheet)
        self.button_analyze.setObjectName("button_analyze")
        self.button_draw.setObjectName("button_draw")
        self.button_instructions.setObjectName("button_instructions")
        self.button_generate_data.setObjectName("button_generate_data")
        self.button_train_model.setObjectName("button_train_model")
        self.button_infer.setObjectName("button_infer")

    def open_analyze_dialog(self):
        dialog = AnalyzeDialog(self)
        dialog.exec_()

    def open_train_dialog(self):
        dialog = TrainDialog(self)
        dialog.exec_()

    def plot_solution(self):
        if self.solution:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.solution.t, self.solution.y[0], label='y(t)')
            ax.set_xlabel('t')
            ax.set_ylabel('y(t)')
            ax.set_title('ODE Solution')
            ax.legend()
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "错误", "请先解析并求解常微分方程。")

    def show_instructions(self):
        instructions = """
        使用说明：
        1. 点击“解析”按钮，输入函数表达式、常系数、阶数、初始条件、时间区间和步数。
        2. 点击“生成数据集”按钮生成数据。
        3. 点击“训练模型”按钮，输入模型参数并训练神经网络。
        4. 点击“绘制曲线”按钮查看ODE解的曲线。
        5. 点击“推理”按钮，使用训练好的模型进行推理并绘制曲线。
        """
        QMessageBox.information(self, "使用说明", instructions)

    def generate_data(self):
        if self.solution:
            self.dataset = CustomDataset(self.solution.t,self.solution.y[0])
            data = np.column_stack((self.solution.t, self.solution.y[0]))
            np.savetxt('data.csv', data, delimiter=',', header='t,y', comments='')
            QMessageBox.information(self, "完成", "数据集已生成并保存在data.csv。")
        else:
            QMessageBox.warning(self, "错误", "请先解析并求解常微分方程。")

    def infer_model(self):
        if self.model is None:
            QMessageBox.warning(self, "错误", "请先训练模型。")
            return

        with torch.no_grad():
            x = torch.tensor(self.dataset.x.reshape(-1, 1), dtype=torch.float32).to("cuda")
            y_pred = self.model(x).cpu().numpy()
            y_pred = y_pred * np.std(self.solution.y[0]) + np.mean(self.solution.y[0])
            data = np.column_stack((self.solution.t, y_pred))
            np.savetxt('data_mlp.csv', data, delimiter=',', header='t,y', comments='')
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.solution.t, self.solution.y[0], label='ODE Solution')
        ax.plot(self.solution.t, y_pred, label='Model Prediction')
        ax.set_xlabel('t')
        ax.set_ylabel('y')
        ax.set_title('ODE Solution vs Model Prediction')
        ax.legend()
        self.canvas.draw()
        QMessageBox.information(self, "完成", "黄线为模型预测")

class AnalyzeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("解析并求解")

        layout = QFormLayout()

        self.function_text = QLineEdit()
        self.coefficients_text = QLineEdit()
        self.order_text = QLineEdit()
        self.initial_conditions_text = QLineEdit()
        self.t_span_text = QLineEdit()
        self.t_eval_text = QLineEdit()

        layout.addRow("函数表达式", self.function_text)
        layout.addRow("常系数 (逗号分隔)", self.coefficients_text)
        layout.addRow("阶数 n", self.order_text)
        layout.addRow("初始条件 (逗号分隔)", self.initial_conditions_text)
        layout.addRow("时间区间 t_span (起始时间, 结束时间)", self.t_span_text)
        layout.addRow("时间步数 t_eval", self.t_eval_text)

        self.solve_button = QPushButton("解析并求解")
        self.solve_button.setObjectName("solve_button")
        self.solve_button.clicked.connect(self.solve_ode)
        layout.addWidget(self.solve_button)

        self.setLayout(layout)

    def solve_ode(self):
        try:
            f_x_text = "f(x,y) = "+self.function_text.text().strip()
            coefficients = list(map(float, self.coefficients_text.text().split(',')))
            n = int(self.order_text.text())
            initial_conditions = list(map(float, self.initial_conditions_text.text().split(',')))
            t_span = tuple(map(float, self.t_span_text.text().split(',')))
            t_eval = int(self.t_eval_text.text())

            solver = ODESolver(f_x_text, coefficients, n)
            self.parent().solution = solver.solve(initial_conditions, t_eval, t_span)
            QMessageBox.information(self, "成功", "常微分方程求解成功。")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))


class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("训练模型")

        layout = QFormLayout()

        self.layer_num_text = QLineEdit()
        self.layer_links_text = QLineEdit()
        self.num_epoch_text = QLineEdit()
        self.batch_size_text = QLineEdit()
        self.lr_text = QLineEdit()

        layout.addRow("层数", self.layer_num_text)
        layout.addRow("每层神经元数量 (逗号分隔)", self.layer_links_text)
        layout.addRow("训练轮数", self.num_epoch_text)
        layout.addRow("批次大小", self.batch_size_text)
        layout.addRow("学习率", self.lr_text)

        self.train_button = QPushButton("训练")
        self.train_button.setObjectName("train_button")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def train_model(self):
        # 清空目录中的文件
        def clear_directory(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        try:
            if self.parent().solution is None:
                QMessageBox.warning(self, "错误", "请先解析并求解常微分方程。")
                return

            layer_num = int(self.layer_num_text.text())
            layer_links = list(map(int, self.layer_links_text.text().split(',')))
            num_epoch = int(self.num_epoch_text.text())
            batch_size = int(self.batch_size_text.text())
            lr = float(self.lr_text.text())
            savedir = "./save/"
            # 清空 savedir 目录中的文件
            clear_directory(savedir)

            train_model(self.parent().dataset, layer_num, layer_links, num_epoch, batch_size, 0, lr, savedir)

            # 加载训练好的模型
            self.parent().model = create_model(layer_num, layer_links)
            model_path = savedir + f"model_epoch_{num_epoch}.pth"
            if not os.path.exists(model_path):
                model_path = savedir + "best_model.pth"
            self.parent().model.load_state_dict(torch.load(model_path))
            self.parent().model = self.parent().model.to("cuda")

            QMessageBox.information(self, "完成", "模型训练完成")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())