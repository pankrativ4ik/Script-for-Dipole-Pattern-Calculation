import numpy as np
import matplotlib.pyplot as plt

class DipolePattern:
    def __init__(self, freq_ghz=1.5, two_l_over_lambda=1.5, data_file="data for 2.txt"):
        self.c = 299_792_458.0
        self.f = freq_ghz * 1e9
        self.two_l_over_lambda = two_l_over_lambda
        self.lmbda = self.c / self.f
        self.l = 0.5 * two_l_over_lambda * self.lmbda
        self.k = 2 * np.pi / self.lmbda
        self.kl = self.k * self.l
        self.data_file = data_file

    def E_theta_mag(self, theta):
        """|Ėθ(θ)| ∝ | sin(kl) · [cos(kl cosθ) − cos(kl)] / sinθ |"""
        th = np.array(theta, dtype=float)
        num = np.cos(self.kl * np.cos(th)) - np.cos(self.kl)
        den = np.sin(th)
        eps = 1e-12
        den_safe = np.where(np.abs(den) < eps, eps, den)

        # модуль |E_θ(θ)|
        E_abs = np.abs(np.sin(self.kl) * (num / den_safe))
        return E_abs

    def compute_analytic_D(self, n_theta=1801):
        """формулы"""
        # сетка углов θ ∈ [0, π]
        theta = np.linspace(0.0, np.pi, n_theta)

        # |E_θ(θ)|
        E = self.E_theta_mag(theta)

        #F(θ) = |E_θ(θ)| / max(|E_θ(θ)|)
        Emax = E.max()
        if Emax == 0:
            raise RuntimeError("Максимум поля равен нулю")
        F = E / Emax

        #D_max = 2 / ∫_0^π F^2(θ) sinθ dθ
        integrand = F**2 * np.sin(theta)
        integral = np.trapz(integrand, theta)
        Dmax = 2.0 / integral

        #D(θ) = F^2(θ) · D_max
        D_theta = F**2 * Dmax

        return theta, D_theta, Dmax

    def read_simulation(self):
        """чтение результатов электродинамического моделирования"""
        thetas = []
        dBi = []

        with open(self.data_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("theta") or line.startswith("-"):
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

    def plot_all(self, theta_a, D_a, theta_s, D_s_lin, D_s_dBi):
        """построение графиков"""
        eps = 1e-12
        D_a_db = 10 * np.log10(np.maximum(D_a, eps))

        fig = plt.figure(figsize=(12, 10))

        # график 1
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(np.rad2deg(theta_a), D_a, label="Аналитика", lw=2)
        ax1.plot(np.rad2deg(theta_s), D_s_lin, "o", ms=3, label="Моделирование")
        ax1.set_xlabel("θ, град")
        ax1.set_ylabel("D")
        ax1.set_title("D(θ) — декарт, линейная шкала")
        ax1.grid(True)
        ax1.legend()

        # график 2
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(np.rad2deg(theta_a), D_a_db, label="Аналитика", lw=2)
        ax2.plot(np.rad2deg(theta_s), D_s_dBi, "o", ms=3, label="Моделирование")
        ax2.set_xlabel("θ, град")
        ax2.set_ylabel("D, dBi")
        ax2.set_title("D(θ) — декарт, dB")
        ax2.grid(True)
        ax2.legend()

        # график 3
        ax3 = fig.add_subplot(2, 2, 3, projection="polar")
        ax3.plot(theta_a, D_a, label="Аналитика", lw=2)
        ax3.plot(theta_s, D_s_lin, "o", ms=3, label="Моделирование")
        ax3.set_title("Полярная диаграмма (лин.)", va="bottom")
        ax3.legend(loc="upper right")

        # график 4
        ax4 = fig.add_subplot(2, 2, 4, projection="polar")
        ax4.plot(theta_a, D_a_db, label="Аналитика", lw=2)
        ax4.plot(theta_s, D_s_dBi, "o", ms=3, label="Моделирование")
        ax4.set_title("Полярная диаграмма (dB)", va="bottom")
        ax4.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

def main():
    dp = DipolePattern(freq_ghz=1.5, two_l_over_lambda=1.5)
    theta_a, D_a, Dmax = dp.compute_analytic_D()
    theta_s, D_s_lin, D_s_dBi = dp.read_simulation()
    print(f"D_max = {Dmax:.6f} ({10*np.log10(Dmax):.3f} dBi)")
    dp.plot_all(theta_a, D_a, theta_s, D_s_lin, D_s_dBi)


if __name__ == "__main__":
    main()
