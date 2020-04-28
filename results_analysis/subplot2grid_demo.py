import matplotlib.pyplot as plt
# plt.figure(figsize=(7.4861,5.54))
# ax1 = plt.subplot2grid((2,6), (0,0), colspan=2,aspect='equal')
# ax2 = plt.subplot2grid((2,6), (0,2), colspan=2,aspect='equal')
# ax3 = plt.subplot2grid((2,6), (0,4), colspan=2,aspect='equal')
# ax4 = plt.subplot2grid((2,6), (1,1), colspan=2,aspect='equal')
# ax5 = plt.subplot2grid((2,6), (1,3), colspan=2,aspect='equal')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(7.4861,6.54))
# ax1 = plt.subplot2grid((3,4), (0,0), colspan=2,)
# ax2 = plt.subplot2grid((3,4), (0,2), colspan=2,)
# ax3 = plt.subplot2grid((3,4), (1,0), colspan=2,)
# ax4 = plt.subplot2grid((3,4), (1,2), colspan=2,)
# ax5 = plt.subplot2grid((3,4), (2,1), colspan=2,)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(3.54,3.54))
# ax1 = plt.subplot2grid((2,2), (0,0), colspan=1,)
# ax2 = plt.subplot2grid((2,2), (0,1), colspan=1,)
# ax3 = plt.subplot2grid((2,2), (1,0), colspan=2,)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(3.54,3.54))
# ax1 = plt.subplot2grid((3,2), (0,0), colspan=1,)
# ax2 = plt.subplot2grid((3,2), (0,1), colspan=1,)
# ax3 = plt.subplot2grid((3,2), (1,0), colspan=1,)
# ax4 = plt.subplot2grid((3,2), (1,1), colspan=1,)
# ax5 = plt.subplot2grid((3,2), (2,0), colspan=2,)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(3.54,3.54))
# ax1 = plt.subplot2grid((5,2), (0,0), colspan=1,)
# ax2 = plt.subplot2grid((5,2), (0,1), colspan=1,)
# ax3 = plt.subplot2grid((5,2), (1,0), colspan=1,)
# ax4 = plt.subplot2grid((5,2), (1,1), colspan=1,)
# ax5 = plt.subplot2grid((5,2), (2,0), colspan=1,)
# ax6 = plt.subplot2grid((5,2), (2,1), colspan=1,)
# ax7 = plt.subplot2grid((5,2), (3,0), colspan=1,)
# ax8 = plt.subplot2grid((5,2), (3,1), colspan=1,)
# ax9 = plt.subplot2grid((5,2), (4,0), colspan=2,)
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(7.4861,4.48))
ax1  = plt.subplot2grid((5,360), (0,0), colspan=30,)#ssa
ax2  = plt.subplot2grid((5,360), (0,30), colspan=30,)
ax3  = plt.subplot2grid((5,360), (0,60), colspan=30,)
ax4  = plt.subplot2grid((5,360), (0,90), colspan=30,)
ax5  = plt.subplot2grid((5,360), (0,120), colspan=30,)
ax6  = plt.subplot2grid((5,360), (0,150), colspan=30,)
ax7  = plt.subplot2grid((5,360), (0,180), colspan=30,)
ax8  = plt.subplot2grid((5,360), (0,210), colspan=30,)
ax9  = plt.subplot2grid((5,360), (0,240), colspan=30,)
ax10 = plt.subplot2grid((5,360), (0,270), colspan=30,)
ax11 = plt.subplot2grid((5,360), (0,300), colspan=30,)
ax12 = plt.subplot2grid((5,360), (0,330), colspan=30,)
ax13 = plt.subplot2grid((5,360), (1,0), colspan=40,)#eemd
ax14 = plt.subplot2grid((5,360), (1,40), colspan=40,)
ax15 = plt.subplot2grid((5,360), (1,80), colspan=40,)
ax16 = plt.subplot2grid((5,360), (1,120), colspan=40,)
ax17 = plt.subplot2grid((5,360), (1,160), colspan=40,)
ax18 = plt.subplot2grid((5,360), (1,200), colspan=40,)
ax19 = plt.subplot2grid((5,360), (1,240), colspan=40,)
ax20 = plt.subplot2grid((5,360), (1,280), colspan=40,)
ax21 = plt.subplot2grid((5,360), (1,320), colspan=40,)
ax22 = plt.subplot2grid((5,360), (2,0), colspan=45,)#vmd
ax23 = plt.subplot2grid((5,360), (2,45), colspan=45,)
ax24 = plt.subplot2grid((5,360), (2,90), colspan=45,)
ax25 = plt.subplot2grid((5,360), (2,135), colspan=45,)
ax26 = plt.subplot2grid((5,360), (2,180), colspan=45,)
ax27 = plt.subplot2grid((5,360), (2,225), colspan=45,)
ax28 = plt.subplot2grid((5,360), (2,270), colspan=45,)
ax29 = plt.subplot2grid((5,360), (2,315), colspan=45,)
ax30 = plt.subplot2grid((5,360), (3,0), colspan=72,)#modwt
ax31 = plt.subplot2grid((5,360), (3,72), colspan=72,)
ax32 = plt.subplot2grid((5,360), (3,144), colspan=72,)
ax33 = plt.subplot2grid((5,360), (3,216), colspan=72,)
ax34 = plt.subplot2grid((5,360), (3,288), colspan=72,)
ax35 = plt.subplot2grid((5,360), (4,0), colspan=120,)#DWT
ax36 = plt.subplot2grid((5,360), (4,120), colspan=120,)
ax37 = plt.subplot2grid((5,360), (4,240), colspan=120,)
axs=[
    ax1 ,ax2 ,ax3 ,ax4 ,ax5 ,ax6 ,ax7 ,ax8 ,ax9 ,ax10,ax11,ax12,
    ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20,ax21,
    ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29,
    ax30,ax31,ax32,ax33,ax34,
    ax35,ax36,ax37,
]
plt.tight_layout()
plt.show()





