import matplotlib.pyplot as plt
import matplotlib as mpl
FONTPATH = "fonts/SourceHanSerifSC-SemiBold.otf"
mpl.font_manager.fontManager.addfont(FONTPATH)
plt.rcParams['font.family'] = ['Source Han Serif SC']


mcm = McCallModel()
V, U = solve_mccall_model(mcm)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(mcm.w_vec, V, 'b-', lw=2, alpha=0.7, label='$V$')
ax.plot(mcm.w_vec, [U]*len(mcm.w_vec), 'g-', lw=2, alpha=0.7, label='$U$')
ax.set_xlim(min(mcm.w_vec), max(mcm.w_vec))
ax.legend(loc='upper left')
ax.grid()

plt.show()
