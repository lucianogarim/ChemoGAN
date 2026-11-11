"""Central experiment configuration for ChemoGAN (revision 1, CAGEO-D-26-00775).

Every constant used by training, constraint evaluation (PVR), rejection sampling
and downstream regression is defined here, so that the manuscript, the code and
the reported experiments share a single source of truth.
"""

DATA_DIR = "wells"

MNEMONICS = ['DEPTH', 'HCAL', 'DWAL', 'DWCA', 'DWFE', 'DWSI', 'RHOB', 'GR', 'NPHI', 'PE', 'DT',
             'NMRE_FINAL', 'NMRFL_FINAL', 'NMRT_FINAL', 'HFK', 'HTHO', 'HURA', 'T2LM']

# Reservoir (Barra Velha) interval per well. Only wells listed here are used (21 wells).
# Note: in the first submission, 7-SPH-8-SPS was loaded without a depth filter and silently
# entered the training set; it is now excluded so that the dataset matches the 21 wells
# described in the manuscript.
DEPTH_FILTERS = {
    '7-SPH-3-SPS_BASE.las': (5037, 5450),
    '7-SPH-20D-SPS_BASE.las': (5050, 5277), '7-SPH-15D-SPS_BASE.las': (5430, 5722),
    '1-SPS-96-SP_BASE.las': (5037, 5483), '3-SPS-69-SP_BASE.las': (4964, 5175),
    '3-SPS-82A-SP_BASE.las': (5076, 5494), '7-SPH-1-SPS_BASE.las': (4995, 5357),
    '7-SPH-4D-SPS_BASE.las': (5150, 5400), '7-SPH-5-SPS_BASE.las': (5053, 5190),
    '7-SPH-6-SPS_BASE.las': (5000, 5368), '7-SPH-7D-SPS_BASE.las': (5220, 5487),
    '7-SPH-14D-SPS_BASE.las': (5193, 5460), '7-SPH-16D-SPS_BASE.las': (5145, 5285),
    '7-SPH-17-SPS_BASE.las': (5031, 5200), '7-SPH-22-SPS_BASE.las': (4980, 5030),
    '8-SPH-9-SPS_BASE.las': (5128, 5356), '8-SPH-11-SPS_BASE.las': (5091, 5280),
    '8-SPH-21D-SPS_BASE.las': (5117, 5496), '8-SPH-23-SPS_BASE.las': (5105, 5478),
    '9-SPS-77A-SP_BASE.las': (5023, 5339), '9-SPS-97-SP_BASE.las': (5138, 5190),
}

# Blind test wells: never used for fitting, scaling, tuning or model selection.
BLIND_WELLS = ['7-SPH-3-SPS_BASE.las', '9-SPS-77A-SP_BASE.las', '9-SPS-97-SP_BASE.las']

# Validation wells: held out from generator training and used ONLY for model-selection
# decisions (epoch budget, generator variants); never for reporting.
VAL_WELLS = ['7-SPH-6-SPS_BASE.las', '8-SPH-23-SPS_BASE.las', '7-SPH-17-SPS_BASE.las']

PREDICTORS = ['GR', 'PE', 'DT', 'NMRE_FINAL', 'NMRFL_FINAL', 'NMRT_FINAL',
              'HFK', 'HTHO', 'HURA', 'T2LM', 'RHOB', 'NPHI']

WINDOW_T = 40          # window length (samples, 0.1524 m spacing -> ~6.1 m)
STRIDE = WINDOW_T // 4  # s = T/4 = 10
LATENT_Z = 100
# Revision 1: batch 128 and lr 2e-4 (standard DCGAN setting). With batch 512 each epoch has
# only ~5 generator updates, which is why the first submission needed 3000 epochs. Note that
# the correlation penalty is computed per window (over T), so its variance does not depend
# on the batch size.
BATCH_SIZE = 128
GAN_LR = 2e-4
ISOLATION_CONTAMINATION = 0.03

# ---------------------------------------------------------------------------------
# Petrophysical constraints (shared by the training loss, the PVR and rejection).
# Hard constraint: NMR porosity chain phi_T >= phi_E >= phi_FL (point-wise).
# Soft constraints: window-wise Pearson correlations with expected sign s and |r| >= tau.
# ---------------------------------------------------------------------------------
TAU = 0.5
NMR_TOL = 1e-4  # porosity units (v/v)
NMR_COLS = ('NMRT_FINAL', 'NMRE_FINAL', 'NMRFL_FINAL')
COMMON_PAIRS = [('NPHI', 'RHOB', -1), ('NPHI', 'DT', +1)]
TARGET_PAIRS = {'DWSI': [('PE', 'DWSI', -1)], 'DWCA': [('PE', 'DWCA', +1)]}


def features_for(target):
    return PREDICTORS + [target]


def pairs_for(target, include_target_pair=True):
    return COMMON_PAIRS + (TARGET_PAIRS[target] if include_target_pair else [])


# Default penalty weights (lambda) for ChemoGAN.
LAMBDAS = {'tv': 1.0, 'nmr': 2.0, 'corr': 2.0}
# Revision 1: NMR ordering enforced by construction (gan_lib.NMROrderLayer) in ChemoGAN.
NMR_HARD = True
