import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loaded_numpy_data_dict = np.load(
    "/media/wenuka/New Volume-G/01.FYP/Tool_1/Simple MIL/AttentionDeepMIL/attention/attention_1image_par_bag.npy", allow_pickle=True
).item()

att = loaded_numpy_data_dict['Lung_Dx-G0051&11-04-2010-PET03CBMWholebodyFirstHead_Adult-54846&10.000000-Thorax_1.0_B70f-83663_patch']
att_reshaped = att.reshape(8, 8, 8)
name = 'Lung_Dx-G0051&11-04-2010-PET03CBMWholebodyFirstHead_Adult-54846&10.000000-Thorax_1.0_B70f-83663_patch'
for i in range(att_reshaped.shape[2]):
        slice = att_reshaped[:, :, i]
        slice = slice.T
        
        # Plotting the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            slice.cpu(),
            cmap="viridis",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Attention"},
        )
        ax.invert_yaxis()
        plt.title("Attention Across Patches")
        plt.savefig(
            f"/media/wenuka/New Volume-G/01.FYP/Tool_1/Simple MIL/AttentionDeepMIL/attention/maps/{name}_slice{i}.png"
        )