"""
Вариант: f=1.5 GHz, 2l/λ = 1.5 (=> l = 0.75 * λ)
"""

import numpy as np
import matplotlib.pyplot as plt

class DipolePattern:
    def __init__(self, freq_ghz=1.5, two_l_over_lambda=1.5, data_file='data for 2.txt'):
        self.c = 299792458.0
        self.f = freq_ghz * 1e9
        self.two_l_over_lambda = two_l_over_lambda
        # длина волны:
        self.lmbda = self.c / self.f
        # полная длина 2l = two_l_over_lambda * lambda -> плечо l:
        self.l = 0.5 * two_l_over_lambda * self.lmbda
        # волновое число
        self.k = 2 * np.pi / self.lmbda
        self.kl = self.k * self.l  # k*l
        self.data_file = data_file

    def E_theta_mag(self, theta):
        """
        Формула: | sin(kl) * [cos(kl cosθ) - cos(kl)] / sinθ |
        """
        th = np.array(theta, dtype=float)
        num = np.cos(self.kl * np.cos(th)) - np.cos(self.kl)
        den = np.sin(th)
        eps = 1e-12
        den_safe = np.where(np.abs(den) < eps, eps, den)
        val = np.abs(np.sin(self.kl) * (num / den_safe))
        return val

    def compute_analytic_D(self, n_theta=1801):
        # сетка theta от 0 до pi
        theta = np.linspace(0.0, np.pi, n_theta)
        E = self.E_theta_mag(theta)
        # нормировка F = |E| / max(|E|)
        Emax = E.max()
        if Emax == 0:
            raise RuntimeError("Максимум поля получился нулевым.")
        F = E / Emax
        # вычисление Dmax через интеграл: Dmax = 2 / ∫_0^π F^2 sinθ dθ
        integrand = F**2 * np.sin(theta)
        integral = np.trapz(integrand, theta)
        Dmax = 2.0 / integral
        # D(theta) = F^2 * Dmax
        D_theta = F**2 * Dmax
        return theta, D_theta, Dmax

    def read_simulation(self):
        """
        Читает файл моделирования. Ожидается текстовый файл со строками:
        Theta [deg.]  Phi [deg.]  Abs(Dir.)[dBi]  ...
        Возвращает theta_rad, D_sim_linear, D_sim_dBi
        """
        thetas = []
        dBi = []
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('theta') or line.startswith('-') or line.startswith('Theta'):
                    continue
                parts = line.split()
                try:
                    th = float(parts[0])
                    val_dbi = float(parts[2])
                except Exception:
                    continue
                thetas.append(th)
                dBi.append(val_dbi)
        theta_rad = np.deg2rad(np.array(thetas))
        D_sim_lin = 10.0 ** (np.array(dBi) / 10.0)
        return theta_rad, D_sim_lin, np.array(dBi)

    def plot_all(self, theta_analytic, D_analytic, Dmax, theta_sim, D_sim_lin, D_sim_dBi):
        eps = 1e-12
        D_analytic_db = 10.0 * np.log10(np.maximum(D_analytic, eps))
        D_sim_db = D_sim_dBi
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(2,2,1)
        ax1.plot(np.rad2deg(theta_analytic), D_analytic, label='Аналитика', lw=2)
        ax1.plot(np.rad2deg(theta_sim), D_sim_lin, 'o', ms=3, label='Моделирование (сим.)')
        ax1.set_xlabel('θ, deg')
        ax1.set_ylabel('D (разы)')
        ax1.set_title('D(θ) — декарт (линейная шкала)')
        ax1.grid(True)
        ax1.legend()
        ax2 = fig.add_subplot(2,2,2)
        ax2.plot(np.rad2deg(theta_analytic), D_analytic_db, label='Аналитика', lw=2)
        ax2.plot(np.rad2deg(theta_sim), D_sim_db, 'o', ms=3, label='Моделирование (сим.)')
        ax2.set_xlabel('θ, deg')
        ax2.set_ylabel('D (dBi)')
        ax2.set_title('D(θ) — декарт (dB)')
        ax2.grid(True)
        ax2.legend()
        ax3 = fig.add_subplot(2,2,3, projection='polar')
        ax3.plot(theta_analytic, D_analytic, label='Аналитика', lw=2)
        ax3.plot(theta_sim, D_sim_lin, 'o', ms=3, label='Моделирование (сим.)')
        ax3.set_title('D(θ) — поляр (линейная шкала)', va='bottom')
        ax3.legend(loc='upper right')
        ax4 = fig.add_subplot(2,2,4, projection='polar')
        ax4.plot(theta_analytic, D_analytic_db, label='Аналитика', lw=2)
        ax4.plot(theta_sim, D_sim_db, 'o', ms=3, label='Моделирование (сим.)')
        ax4.set_title('D(θ) — поляр (dB)', va='bottom')
        ax4.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

def main():
    data_path = 'data for 2.txt'
    dp = DipolePattern(freq_ghz=1.5, two_l_over_lambda=1.5, data_file=data_path)
    theta_a, D_a, Dmax = dp.compute_analytic_D(n_theta=1801)
    theta_s, D_s_lin, D_s_dBi = dp.read_simulation()
    Dmax_db = 10.0 * np.log10(Dmax)
    print(f"Вариант: f = {dp.f/1e9:.3f} GHz, 2l/λ = {dp.two_l_over_lambda:.3f}, l = {dp.l:.6e} m, kl = {dp.kl:.6f}")
    print(f"D_max (аналитически): {Dmax:.6f} раз; {Dmax_db:.3f} dBi")
    dp.plot_all(theta_a, D_a, Dmax, theta_s, D_s_lin, D_s_dBi)

if __name__ == '__main__':
    main()