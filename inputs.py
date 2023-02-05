import numpy as np
from neat.fields import StellnaQS
# Number of aspect ratio to choose
n_minor_radius = 10 # number of minor radii for scan
r_min = 0.05 # minimum minor radius to use in meters (maximum radius usually 1.7044/1.3=1.31)
rminor_factor=1.3 # ratio between max minor radius to use and ARIES_CS minor radius
maximum_s_gyronimo = 0.95 # maximum s value to integrate in gyronimo using vmectrace
# Particle and Integration Properties
r_initial = 0.03  # meters
theta_initial = np.pi/4  # initial Boozer poloidal angle
varphi_initial = np.pi/4  # initial cylindrical toroidal angle
Lambda = 0.4  # = mu * B0 / energy
nsamples = 1000  # resolution in time
tfinal = 1e-5  # seconds
# Size of equilibrium
Rmajor_ARIES = 7.7495
Aminor_ARIES=1.7044
b0_ARIES=5.3267
Psi_ARIES=51.4468
r_ARIES = Aminor_ARIES/rminor_factor # true r = np.sqrt(Psi_ARIES/b0_ARIES/(np.pi))
minor_radius_array = np.logspace(np.log10(r_min),np.log10(r_ARIES),n_minor_radius)
# minor_radius_array = np.linspace(r_min,r_ARIES,n_minor_radius)
# Near-Axis Magnetic Field
constant_b20 = False  # use a constant B20 (mean value) or the real function
rc = np.array([1.0e+0, 5.022414262333900481e-02, 2.464859129049914218e-03, 1.119266286864647027e-04, 4.182627242567557729e-06, 2.073762398158625685e-07])
zs = np.array([0.0e+0,-5.121909533773784384e-02,-2.422089652869440467e-03,-1.137537415304500467e-04,-4.344635778503562632e-06,-3.008066549137089361e-07])
B2c = 2.1121434634154563
nfp = 3
eta_bar = 0.9256884588145107
field_nearaxis = StellnaQS(rc=rc*Rmajor_ARIES, zs=zs*Rmajor_ARIES, etabar=eta_bar/Rmajor_ARIES, B2c=B2c*(b0_ARIES/Rmajor_ARIES/Rmajor_ARIES), B0=b0_ARIES, nfp=nfp, order='r3', nphi=111)