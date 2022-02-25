import matplotlib.pyplot as plt 

xx = [0,   10,   25,  50,  100,             200, 300, 400,  500]
vae = [0.,  0.07,  0.17, 0.24, 0.4,            0.56, 0.68, 0.86, 0.96]
vae2 = [itv+0.2 for itv in vae]
vae_bar = [0, 0.075, 0.064, 0.055, 0.045,     0.036, 0.025, 0.02, 0.01]

plt.figure()
_, caps, _, = plt.errorbar(xx, vae, yerr=vae_bar, ecolor='green', color='green', fmt='-s', markersize=4, capsize=3, label='Lasso')
# for cap in caps:
#     cap.set_markeredgewidth(1)

plt.xlabel('Number of measurements (m)', fontsize=16)
plt.ylabel('Cosine Similarity', fontsize=16)
# plt.xticks(xx, xx)    
# plt.legend(loc='lower right', fontsize=20)  
plt.savefig('mnist_cos.pdf')



fig, ax = plt.subplots() 
ax.errorbar(xx,vae,vae_bar, ecolor='black', color='black', fmt='-s', markersize=4, capsize=3, label='Power')
ax.errorbar(xx,vae2,vae_bar, ecolor='blue', color='blue', fmt='-s', markersize=4, capsize=3, label='TPower')

ax.legend(loc = 'lower right')
ax.set_xlabel('Number of Measurements(m)')
ax.set_ylabel('Cosine Similarity')
ax.grid(True)
plt.show()    