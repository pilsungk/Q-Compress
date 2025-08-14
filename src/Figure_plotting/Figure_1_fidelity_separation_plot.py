import matplotlib.pyplot as plt
import numpy as np

# Figure setting
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.family'] = 'DejaVu Sans'

# Figure generation
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4,12))

# colors for paper
color_robust = '#2E86AB'
color_fragile = '#E63946'
color_gap = '#E0E0E0'

# 10 qubits data
# Fragile circuits (22)
fragile_10_indices = [39, 60, 30, 13, 51, 50, 97, 40, 91, 78, 15, 95, 86, 10, 8, 27, 99, 85, 43, 35, 32, 73]
fragile_10_values = [0.4507158632, 0.4526539067, 0.5358421612, 0.5439146196, 0.593705538, 0.6019912741, 
                     0.6085430307, 0.6532158607, 0.6634430478, 0.6720071862, 0.6875849481, 0.6934127434, 
                     0.7532000624, 0.7822578913, 0.7927505326, 0.7998077584, 0.8472131511, 0.8526484015, 
                     0.8618033147, 0.8705395638, 0.8748583724, 0.8982855032]
# Robust circuits (78) 
robust_10_indices = [i for i in range(100) if i not in fragile_10_indices]

# 12 qubits data
# Fragile circuits (28)
fragile_12_indices = [1, 5, 6, 8, 13, 14, 15, 18, 27, 34, 35, 36, 39, 40, 41, 42, 43, 46, 50, 51, 60, 61, 78, 85, 88, 90, 91, 92]
fragile_12_values = [0.7319032859, 0.6291966928, 0.5647876366, 0.8724165282, 0.8747877345, 0.7435713726, 
                     0.8338483589, 0.8612868452, 0.6577186575, 0.7397528526, 0.7039097471, 0.8832215428, 
                     0.4325223569, 0.8438381246, 0.7335112149, 0.6291805478, 0.7588158129, 0.5572722399, 
                     0.7534667477, 0.6359326366, 0.4507241998, 0.8262230525, 0.7341649391, 0.7817381564, 
                     0.8572907575, 0.5385578952, 0.8430531933, 0.6002105669]
# Robust circuits (72)
robust_12_indices = [i for i in range(100) if i not in fragile_12_indices]

# 14 qubits data
# Fragile circuits (37)
fragile_14_indices = [5, 48, 8, 18, 19, 42, 6, 24, 39, 61, 78, 34, 88, 51, 89, 90, 59, 80, 46, 35, 50, 83, 92, 91, 60, 1, 55, 17, 98, 30, 4, 41, 27, 43, 7, 14, 93]
fragile_14_values = [0.3508652586, 0.4275388622, 0.4420367394, 0.4441064754, 0.4673886032, 0.4775288991, 
                     0.48884005, 0.518595735, 0.5286753334, 0.5410505431, 0.557261352, 0.5926769299, 
                     0.6019310433, 0.6198502734, 0.6355736637, 0.6371177196, 0.6837698002, 0.6964426858, 
                     0.7053081009, 0.7102496474, 0.7160876519, 0.7247075418, 0.738381344, 0.780170969, 
                     0.7872712061, 0.8000549719, 0.801283932, 0.8119294214, 0.8186017899, 0.8266801965, 
                     0.8292660656, 0.836703959, 0.8693369745, 0.8746862515, 0.8766084489, 0.8795669345, 
                     0.8940882095]
# Robust circuits (63)
robust_14_indices = [i for i in range(100) if i not in fragile_14_indices]

# Robust circuits' fidelity values
# 10 qubits robust 
robust_10_values = [0.9225036881, 0.9232862249, 0.9308209069, 0.9359736708, 0.9700256263, 0.9745792276, 
                    0.9830092097, 0.9864374771, 0.986683598, 0.9869849161, 0.9870390448, 0.987054136, 
                    0.9872499388, 0.9873114109, 0.9873341946, 0.9873398525, 0.9873446776, 0.9873640441, 
                    0.9875157096, 0.9875381357, 0.9875673639, 0.9878603687, 0.9880285433, 0.9880700103, 
                    0.9880710969, 0.9881596455, 0.9881639028, 0.9882901825, 0.9883491457, 0.9884486245, 
                    0.9884847406, 0.9885546625, 0.9887325116, 0.9887987696, 0.9890292155, 0.9890294746, 
                    0.9890366105, 0.9890446277, 0.9890528544, 0.9890956344, 0.9891103931, 0.9891310631, 
                    0.9893617793, 0.9893719339, 0.9893779412, 0.989398623, 0.9894812644, 0.9895311502, 
                    0.9895372033, 0.9895513589, 0.989610355, 0.9896401974, 0.9896461527, 0.9896871415, 
                    0.9897175746, 0.9898043463, 0.9899085187, 0.9899363166, 0.9900138005, 0.9900888273, 
                    0.9901727196, 0.990199498, 0.9902201579, 0.9902346072, 0.9903469553, 0.9903565322, 
                    0.990389069, 0.9904354725, 0.9904580236, 0.9905277486, 0.990734904, 0.9908131827, 
                    0.9908444343, 0.990922977, 0.9909554647, 0.9912313195, 0.991311377, 0.992550217]

# 12 qubits robust 
robust_12_values = [0.9174957378, 0.9811001684, 0.9847642331, 0.9830578009, 0.9824729238, 0.9845090529, 
                    0.9849707529, 0.983892408, 0.9865654503, 0.982167167, 0.9853768984, 0.9843186152, 
                    0.9821022828, 0.9838819849, 0.983554888, 0.9891824133, 0.9804884545, 0.9809866314, 
                    0.9825618565, 0.9819285037, 0.9143168822, 0.982009377, 0.9848116088, 0.9853748422, 
                    0.9849349260, 0.9846745114, 0.9806429145, 0.9851072354, 0.9847087117, 0.9851333529, 
                    0.9639439134, 0.9852180691, 0.9875400234, 0.9879919853, 0.9862882762, 0.9856476632, 
                    0.9836776458, 0.9818356275, 0.9858914168, 0.9811286301, 0.9861963598, 0.9826280728, 
                    0.9834512737, 0.9853071870, 0.9858579077, 0.9693533239, 0.9860063830, 0.9844627457, 
                    0.9812913031, 0.9851953262, 0.9850017883, 0.9839389993, 0.9844069099, 0.9861013206, 
                    0.9864007060, 0.9837466048, 0.9842114293, 0.9827096372, 0.9814112832, 0.9842617861, 
                    0.9800230420, 0.9847708391, 0.9835459610, 0.9848147229, 0.9664217479, 0.9843395630, 
                    0.9849433793, 0.9285779889, 0.9811627050, 0.9833809943, 0.9861836026, 0.9575015335]

# 14 qubits robust 
robust_14_values = [0.9095366426, 0.911642954, 0.915178501, 0.9179052217, 0.9235489517, 0.9354353659, 
                    0.9365078082, 0.957951354, 0.9719624522, 0.9751511701, 0.9761916171, 0.9762078504, 
                    0.9762247559, 0.9762831481, 0.9764408021, 0.9767226166, 0.976753399, 0.976855768, 
                    0.9769864056, 0.9770834552, 0.977156638, 0.9775481201, 0.977719397, 0.9778006817, 
                    0.9778812467, 0.9779145641, 0.978105648, 0.9782642956, 0.9782832183, 0.9783021646, 
                    0.9784360969, 0.9784799851, 0.9786598664, 0.9787523314, 0.978880249, 0.9789283345, 
                    0.979119859, 0.9791246677, 0.979170089, 0.9792847096, 0.979402024, 0.9794063095, 
                    0.9794104038, 0.9795113906, 0.9795237536, 0.9795308689, 0.979702513, 0.9797303362, 
                    0.9797362858, 0.9797446249, 0.9798382104, 0.979921759, 0.979927522, 0.9801915329, 
                    0.9802092701, 0.9803617381, 0.9803798759, 0.9804900734, 0.9807120874, 0.9816978358, 
                    0.9820495771, 0.9825521146, 0.9841611583]

# (a) 10 qubits
all_fidelities_10 = np.zeros(100)
# Robust circuits' fidelity population
for i, idx in enumerate(robust_10_indices):
    all_fidelities_10[idx] = robust_10_values[i]
# Fragile circuits' fidelity population
for i, idx in enumerate(fragile_10_indices):
    all_fidelities_10[idx] = fragile_10_values[i]

# colors
colors_10 = [color_robust if i in robust_10_indices else color_fragile for i in range(100)]

ax1.scatter(range(100), all_fidelities_10, c=colors_10, s=25, alpha=0.6)

# Legend dummy plots
ax1.scatter([], [], color=color_robust, s=25, alpha=0.6, label='Robust')
ax1.scatter([], [], color=color_fragile, s=25, alpha=0.6, label='Fragile')

# Gap area (10 qubits: 0.8983 ~ 0.9225)
ax1.axhspan(0.8983, 0.9225, color=color_gap, alpha=0.7, zorder=0)
ax1.axhline(y=0.9225, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax1.axhline(y=0.8983, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

ax1.set_xlabel('Circuit Index', fontsize=12)
ax1.set_ylabel('Fidelity', fontsize=12)
ax1.set_ylim(0.4, 1.02)
ax1.grid(True, alpha=0.2, linewidth=0.5)
ax1.legend(loc='lower right', frameon=True, fancybox=False)

# Subplot caption
ax1.text(0.5, -0.3, '(a) 10 qubits', transform=ax1.transAxes, 
         ha='center', fontsize=13, fontweight='bold')

# (b) 12 qubits
all_fidelities_12 = np.zeros(100)
# Robust circuits' fidelity population
for i, idx in enumerate(robust_12_indices):
    all_fidelities_12[idx] = robust_12_values[i]
# Fragile circuits' fidelity population
for i, idx in enumerate(fragile_12_indices):
    all_fidelities_12[idx] = fragile_12_values[i]

# colors
colors_12 = [color_robust if i in robust_12_indices else color_fragile for i in range(100)]

ax2.scatter(range(100), all_fidelities_12, c=colors_12, s=25, alpha=0.6)

# Legend dummy plots
ax2.scatter([], [], color=color_robust, s=25, alpha=0.6, label='Robust')
ax2.scatter([], [], color=color_fragile, s=25, alpha=0.6, label='Fragile')

# Gap area (12 qubits: 0.8832 ~ 0.9143)
ax2.axhspan(0.8832, 0.9143, color=color_gap, alpha=0.7, zorder=0)
ax2.axhline(y=0.9143, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax2.axhline(y=0.8832, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

ax2.set_xlabel('Circuit Index', fontsize=12)
ax2.set_ylabel('Fidelity', fontsize=12)
ax2.set_ylim(0.4, 1.02)
ax2.grid(True, alpha=0.2, linewidth=0.5)
ax2.legend(loc='lower right', frameon=True, fancybox=False)

# Subplot caption
ax2.text(0.5, -0.3, '(b) 12 qubits', transform=ax2.transAxes, 
         ha='center', fontsize=13, fontweight='bold')

# (c) 14 qubits
all_fidelities_14 = np.zeros(100)
# Robust circuits' fidelity population
for i, idx in enumerate(robust_14_indices):
    all_fidelities_14[idx] = robust_14_values[i]
# Fragile circuits' fidelity population
for i, idx in enumerate(fragile_14_indices):
    all_fidelities_14[idx] = fragile_14_values[i]

# colors
colors_14 = [color_robust if i in robust_14_indices else color_fragile for i in range(100)]

ax3.scatter(range(100), all_fidelities_14, c=colors_14, s=25, alpha=0.6)

# Legend dummy plots
ax3.scatter([], [], color=color_robust, s=25, alpha=0.6, label='Robust')
ax3.scatter([], [], color=color_fragile, s=25, alpha=0.6, label='Fragile')

# Gap area (14 qubits: 0.8941 ~ 0.9095)
ax3.axhspan(0.8941, 0.9095, color=color_gap, alpha=0.7, zorder=0)
ax3.axhline(y=0.9095, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
ax3.axhline(y=0.8941, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

ax3.set_xlabel('Circuit Index', fontsize=12)
ax3.set_ylabel('Fidelity', fontsize=12)
ax3.set_ylim(0.3, 1.02)
ax3.grid(True, alpha=0.2, linewidth=0.5)
ax3.legend(loc='lower right', frameon=True, fancybox=False)

# Subplot caption
ax3.text(0.5, -0.3, '(c) 14 qubits', transform=ax3.transAxes, 
         ha='center', fontsize=13, fontweight='bold')

fig.subplots_adjust(hspace=0.5)

#plt.tight_layout()

# Save figure
plt.savefig('figure_fidelity_separation_10_12_14_qubits.pdf', dpi=300, bbox_inches='tight')
#plt.savefig('figure_fidelity_separation_10_12_14_qubits.png', dpi=300, bbox_inches='tight')

plt.show()

