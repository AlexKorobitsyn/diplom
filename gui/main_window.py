import tkinter as tk
from tkinter import ttk
from models.solow import SolowModel
from models.RCK import RCKModel
from models.ak import AKModel
import numpy as np
from models.resource_limited import ResourceLimitedModel


class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Модели экономического роста")

        self.model_var = tk.StringVar(value="Solow")

        self.models = {
            'SRL': ResourceLimitedModel(),
            'Solow': SolowModel(),
            'RCK': RCKModel(),
            'AK': AKModel()
        }
        self.params = {
            'alpha': tk.DoubleVar(value=0.5),
            'delta': tk.DoubleVar(value=0.05),
            'n': tk.DoubleVar(value=0.01),
            'g': tk.DoubleVar(value=0.02),
            'rho': tk.DoubleVar(value=0.02),
            'theta': tk.DoubleVar(value=1.5),
            's': tk.DoubleVar(value=0.3),
            'k0_1': tk.DoubleVar(value=1.2),
            'k0_2': tk.DoubleVar(value=7.5),
            'k0_3': tk.DoubleVar(value=31.0),
            'k0_4': tk.DoubleVar(value=42.0)
        }

        self.create_widgets()
        self.model_var.trace_add('write', self.update_ui)
        self.update_ui()

    def create_widgets(self):
        ttk.Label(self.root, text="Модель:").grid(row=0, column=0, sticky='w')
        model_menu = ttk.Combobox(self.root, textvariable=self.model_var,
                                  values=['Solow', 'RCK', 'AK', 'SRL'], state='readonly')
        model_menu.grid(row=0, column=1, sticky='ew')
        self.param_frame = ttk.Frame(self.root)
        self.param_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        ttk.Label(self.root, text="Начальные условия капитала:").grid(row=2, column=0, sticky='w', pady=(10, 0))
        self.k0_frame = ttk.Frame(self.root)
        self.k0_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')

        for i in range(4):
            ttk.Label(self.k0_frame, text=f"k0_{i + 1}:").grid(row=0, column=i * 2, padx=(5, 0))
            ttk.Entry(self.k0_frame, textvariable=self.params[f'k0_{i + 1}'], width=8).grid(row=0, column=i * 2 + 1)
        ttk.Button(self.root, text="Построить график", command=self.run_model).grid(row=4, column=0, columnspan=2,
                                                                                      pady=10)

    def update_ui(self, *args):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        self.entries = {}

        model = self.model_var.get()
        row = 0
        if model == 'SRL':
            ttk.Label(self.param_frame, text='M₀:').grid(row=row, column=0, sticky='w')
            self.entries['M0'] = ttk.Entry(self.param_frame)
            self.entries['M0'].insert(0, '10')
            self.entries['M0'].grid(row=row, column=1)
            row += 1

            for name in ['rho', 'alpha', 'beta', 'mu', 'p0', 'a', 'u_val', 'v_val']:
                ttk.Label(self.param_frame, text=f'{name}:').grid(row=row, column=0, sticky='w')
                self.entries[name] = ttk.Entry(self.param_frame)
                self.entries[name].insert(0, '0.1')
                self.entries[name].grid(row=row, column=1)
                row += 1

            ttk.Label(self.param_frame, text="ρ (Временные предпочтения):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['rho'], width=8).grid(row=row, column=1)
            row += 1

            ttk.Label(self.param_frame, text="θ (Эластичность замещения):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['theta'], width=8).grid(row=row, column=1)
        else:
            ttk.Label(self.param_frame, text="α (Доля капитала):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['alpha'], width=8).grid(row=row, column=1)
            row += 1

            ttk.Label(self.param_frame, text="δ (Амортизация):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['delta'], width=8).grid(row=row, column=1)
            row += 1

            ttk.Label(self.param_frame, text="n (Рост населения):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['n'], width=8).grid(row=row, column=1)
            row += 1

            ttk.Label(self.param_frame, text="g (Тех. прогресс):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['g'], width=8).grid(row=row, column=1)
            row += 1

            if model == 'Solow':
                ttk.Label(self.param_frame, text="s (Норма сбережений):").grid(row=row, column=0, sticky='e')
                ttk.Entry(self.param_frame, textvariable=self.params['s'], width=8).grid(row=row, column=1)
            elif model == 'RCK':
                ttk.Label(self.param_frame, text="ρ (Временные предпочтения):").grid(row=row, column=0, sticky='e')
                ttk.Entry(self.param_frame, textvariable=self.params['rho'], width=8).grid(row=row, column=1)
                row += 1

                ttk.Label(self.param_frame, text="θ (Эластичность замещения):").grid(row=row, column=0, sticky='e')
                ttk.Entry(self.param_frame, textvariable=self.params['theta'], width=8).grid(row=row, column=1)

            elif model == 'AK':
                ttk.Label(self.param_frame, text="s (Норма сбережений):").grid(row=row, column=0, sticky='e')
                ttk.Entry(self.param_frame, textvariable=self.params['s'], width=8).grid(row=row, column=1)


    def run_model(self):
        model_name = self.model_var.get()
        model = self.models[model_name]

        if model_name == 'Solow':
            params = {k: v.get() for k, v in self.params.items()}
            k0s = [params[f'k0_{i + 1}'] for i in range(4)]
            k_star = model.steady_state(**params)

            t_values = []
            k_values = []
            for k0 in k0s:
                t, k = model.simulate(k0, **params)
                t_values.append(t)
                k_values.append(k)

            model.phase_diagram(k0_list=k0s)

        elif model_name == 'RCK':
            params = {k: v.get() for k, v in self.params.items()}
            k0s = [params[f'k0_{i + 1}'] for i in range(4)]

            if hasattr(model, 'trajectories'):
                model.trajectories = []

            for k0 in k0s:
                model.simulate(k0, **params)
            model.phase_diagram()
        elif model_name == 'SRL':
            params = {key: float(entry.get()) for key, entry in self.entries.items()}
            params['t_span'] = (0, 50)
            params['t_eval'] = np.linspace(0, 50, 1000)
            params['x0'] = [0.9, 0.5, 1.0, 0.2, 0.2, 0.2]

            model.simulate(params)
            model.plot_results()

        else:
            params = {k: v.get() for k, v in self.params.items()}
            k0s = [params[f'k0_{i + 1}'] for i in range(4)]
            model.simulate(k0s[0], **params)
            model.phase_diagram()
