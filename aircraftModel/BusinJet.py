"""
BusinJetA(ACtype,MODEL)
FLIGHT Aerodynamic Coefficients of the Business Jet Aircraft, Thrust Model,
and Geometric and Inertial Properties
Business Jet -- Angle-of-Attack-Dependent, Mach-Independent Model
SMOOTHED version of Business Jet A (~LearJet 23/4)
VERSION 2, separating Model Data Tables from AeroModel.m
June 22, 2023
===============================================================
Copyright 2023 by ROBERT F. STENGEL.  All rights reserved.
"""

import numpy as np

class ADB:
    pass

def BusinJetA():
    ACtype = 'BusinJetA'
    MODEL  = 'Alph'

    mSim = 4536         # Reference Mass, kg
    Ixx  = 35926.5      # Roll Moment of Inertia, kg-m^2
    Iyy  = 33940.7      # Pitch Moment of Inertia, kg-m^2
    Izz  = 67085.5      # Yaw Moment of Inertia, kg-m^2
    Ixz  = 3418.17      # Nose-high(low) Product of Inertia, kg-m^2

    # Geometric Properties
    cBar   = 2.14               # Mean Aerodynamic Chord, m
    b      = 10.4               # Wing Span, m
    S      = 21.5               # Reference Area, m^2
    taperw = 0.507              # Wing Taper Ratio
    ARw    = 5.02               # Wing Aspect Ratio
    sweepw = 13 * .01745329     # Wing 1/4-chord sweep angle, rad
    ARh    = 4                  # Horizontal Tail Aspect Ratio
    sweeph = 25 * .01745329     # Horiz Tail 1/4-chord sweep angle, rad
    ARv    = 0.64               # Vertical Tail Aspect Ratio
    sweepv = 40 * .01745329     # Vert Tail 1/4-chord sweep angle, rad
    lvt    = 4.72               # Vert Tail Length, m

    # Maximum Static Thrust @ Sea Level, newtons
    StaticThrust = 26243.2

    # Aerodynamic Data Tables for Business Jet Aircraft
    # Data Tables for High-Angle-of-Attack Business Jet Model
    AlphaTable = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                  25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])  # deg
    AlphaLength = len(AlphaTable)
    AlphaR = np.deg2rad(AlphaTable)
    CosAlphaTable = np.cos(AlphaR)
    CosSqTable = CosAlphaTable**2

    #   Longitudinal
    #   Lift Coefficients
    CLTable = np.array([-0.8681, -0.6870, -0.5348, -0.3536, -0.1797, 0.1101, 0.2899, 0.4638, 0.6449, 0.7971, 0.9783, 1.0507,
               1.1014, 1.1739, 1.2174, 1.2319, 1.1957, 1.1010, 0.9873, 0.9114, 0.8734, 0.8608, 0.8544, 0.8608, 0.8734,
               0.8861, 0.9494, 1.0506, 1.1495, 1.2407, 1.2916, 1.2946, 1.2450, 1.1414, 0.9860, 0.7844, 0.5452, 0.2795, 0])

    CLAlphaTable = np.array([5.1243, 5.0604, 5.1397, 5.4012, 5.8200, 5.8279, 5.3865, 5.1051, 4.8339, 4.5926, 4.2599, 3.6355,
                    3.0656, 2.2637, 0.9322, -0.7641, -2.5316, -3.6899, -3.7080, -2.8643, -1.8214, -0.9642, -0.3171,
                    0.1715, 0.4737, 0.6503, 0.7893, 0.8517, 0.7666, 0.5203, 0.1270, -0.3572, -0.8866, -1.4203, -1.9222,
                    -2.3625, -2.7213, -2.9944, -3.2028])    # per rad

    CLqHatTable = np.array([5.7722, 5.3116, 5.3132, 5.6591, 7.3898, 7.4838, 5.6367, 5.6575, 5.3116, 5.3132, 5.3887, 3.9236,
                   3.9268, 3.6973, 1.8486, -0.6916, -4.1722, -6.6423, -6.0431, -3.6303, -1.6128, -0.6056, 0, 0.6056,
                   0.8064, 0.8075, 1.0486, 1.2756, 1.2118, 0.8569, 0.7789, 0.6951, 0.6059, 0.5121, 0.4145, 0.3136,
                   0.2104, 0.1056, 0.0000])

    CLdETable = 0.5774*CosAlphaTable
    CLdFTable = 0.859*CosAlphaTable
    CLdSTable = 2.5*CLdETable

    # Drag Coefficients
    CDTable = np.array([0.0870, 0.0689, 0.0508, 0.0363, 0.0268, 0.0254, 0.0268, 0.0363,
               0.0508, 0.0689, 0.0870, 0.1051, 0.1233, 0.1450, 0.1668, 0.1885,
               0.2175, 0.2520, 0.2840, 0.3160, 0.3480, 0.3800, 0.4120, 0.4440,
               0.4760, 0.5080, 0.6680, 0.8280, 1.0168, 1.2587, 1.5246, 1.8030,
               2.0806, 2.3433, 2.5773, 2.7697, 2.9093, 2.9879, 3.0000])
    epsilon      = 0.0718              #   Oswald efficiency factor
    CDAlphaTable = np.zeros(AlphaLength)
    CDqHatTable  = np.zeros(AlphaLength)
    CDdETable    = np.zeros(AlphaLength)
    CDdFTable    = 0.057*CosSqTable
    CDdSTable    = np.zeros(AlphaLength)

    # Pitching Moment Coefficients
    CmTable = np.array([0.1900, 0.1520, 0.1140, 0.0760, 0.0380, 0, -0.0380, -0.0760, -0.1140, -0.1520,
              -0.1900, -0.2200, -0.2500, -0.2700, -0.3000, -0.3350, -0.3700, -0.3800, -0.3700, -0.3000,
              -0.2600, -0.2400, -0.2200, -0.2100, -0.2000, -0.1900, -0.1700, -0.1900, -0.2455, -0.3535,
              -0.4396, -0.5327, -0.6061, -0.6515, -0.6898, -0.7196, -0.7400, -0.7502, -0.7500])

    # Pitching Moment Coefficients
    CmAlphaTable = np.array([-0.7276, -0.8165, -0.9511, -1.0199, -1.0528, -1.0671, -1.0714, -1.0671,
                   -1.0514, -1.0184, -1.2681, -1.7389, -1.5871, -1.5155, -1.4839, -1.2634,
                   -0.6732, 0.2292, 1.2433, 1.7303, 1.5126, 1.1917, 0.8909, 0.6045,
                   0.3323, -0.0191, -0.1805, -0.4406, -0.7042, -0.8514, -0.8686, -0.8061,
                   -0.6704, -0.5294, -0.4137, -0.3100, -0.2166, -0.1438, -0.1146]) # per rad
    SMsim  = -CmAlphaTable[5]/CLAlphaTable[5]

    CmqHatTable = np.array([-13.3206, -13.1547, -13.3609, -14.0405, -15.1292, -15.1497, -14.0023, -13.2710,
                  -12.5660, -11.9387, -11.0738, -9.4505, -7.9690, -5.8845, -2.4231, 1.9863,
                  6.5809, 9.5918, 9.6390, 7.4458, 4.7346, 2.5065, 0.8243, -0.4458,
                  -1.2314, -1.6906, -2.0518, -2.2139, -1.9928, -1.3524, -0.3301, 0.9287,
                  2.3047, 3.6921, 4.9969, 6.1415, 7.0742, 7.7840, 8.3257])

    CmdETable = np.array([-1.3841, -1.3858, -1.3893, -1.3814, -1.3731, -1.3749, -1.3730, -1.3812
                -1.3887, -1.3847, -1.3818, -1.3787, -1.3678, -1.3497, -1.3064, -1.2398
                -1.1461, -1.0371, -0.9173, -0.7855, -0.6936, -0.6249, -0.5745, -0.5320
                -0.4908, -0.4597, -0.4378, -0.4142, -0.4360, -0.3964, -0.3574, -0.3175
                -0.2761, -0.2331, -0.1885, -0.1426, -0.0956, -0.0480, 0])

    CmdFTable   =   0.114*CosAlphaTable
    CmdSTable   =     2.5*CmdETable


    # Lateral-Directional
    # Side Force Coefficients
    CYBetaTable = np.array([-0.5777, -0.5919, -0.6035, -0.6139, -0.6216, -0.6261, -0.6203, -0.6106,
                   -0.5967, -0.5781, -0.5502, -0.5156, -0.4801, -0.4398, -0.3976, -0.3740,
                   -0.3527, -0.3045, -0.2284, -0.1463, -0.0589,  0.0222,  0.1005,  0.1644,
                    0.2228,  0.2863,  0.3636,  0.4055,  0.3731,  0.3193,  0.2781,  0.2423,
                    0.2085,  0.1749,  0.1410,  0.1065,  0.0713,  0.0358,  0])

    CYpHatTable = np.zeros(AlphaLength)
    CYrHatTable = np.zeros(AlphaLength)
    CYdATable   = -0.00699*CosSqTable
    CYdRTable   =  0.1574*CosAlphaTable
    CYdASTable  = -0.1*CYdATable

    # Rolling Moment Coefficients
    ClBetaTable = np.array([-0.1777 -0.1744 -0.1713 -0.1686 -0.1664 -0.1649 -0.1642
                   -0.1641 -0.1646 -0.1653 -0.1662 -0.1667 -0.1667 -0.1663 -0.1655
                   -0.1643 -0.1627 -0.1607 -0.1582 -0.1553 -0.1520 -0.1482 -0.1439
                   -0.1390 -0.1335 -0.1274 -0.1207 -0.1133 -0.1052 -0.0966 -0.0873
                   -0.0776 -0.0674 -0.0568 -0.0459 -0.0347 -0.0232 -0.0116 -0.0000])

    ClpHatTable =   (-CLAlphaTable*(1 + 3*taperw)/(12*(1 + taperw)))
    ClrHatTable =   -CLTable * (1 + 3*taperw)/(12*(1 + taperw))
    CldATable = np.array([0.1225, 0.1342, 0.1376, 0.1381, 0.1380, 0.1377, 0.1374, 0.1366,
                 0.1344, 0.1277, 0.1095, 0.0906, 0.0760, 0.0697, 0.0707, 0.0735,
                 0.0759, 0.0761, 0.0759, 0.0756, 0.0752, 0.0747, 0.0740, 0.0733,
                 0.0722, 0.0706, 0.0679, 0.0645, 0.0604, 0.0558, 0.0507, 0.0453,
                 0.0395, 0.0334, 0.0270, 0.0204, 0.0137, 0.0069, 0])
    CldRTable = np.array([0.0092, 0.0105, 0.0125, 0.0147, 0.0165, 0.0175, 0.0165, 0.0146,
                 0.0123, 0.0101, 0.0083, 0.0068, 0.0054, 0.0042, 0.0031, 0.0022,
                 0.0017, 0.0015, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0012,
                 0.0012, 0.0011, 0.0011, 0.0010, 0.0010, 0.0009, 0.0008, 0.0007,
                 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0])
    CldASTable = -0.1*CldATable

    # Yawing Moment Coefficients
    CnBetaTable = np.array([0.1409, 0.1432, 0.1432, 0.1433, 0.1433, 0.1434, 0.1437, 0.1442,
                   0.1452, 0.1471, 0.1488, 0.1555, 0.1706, 0.1948, 0.2194, 0.2313,
                   0.2319, 0.2305, 0.2243, 0.2033, 0.1661, 0.1357, 0.1178, 0.1080,
                   0.1013, 0.0968, 0.0922, 0.0806, 0.0769, 0.0669, 0.0589, 0.0516,
                   0.0446, 0.0375, 0.0302, 0.0228, 0.0153, 0.0077, 0])

    CnpHatTable = CLTable*(1 + 3*taperw)/(12*(1 + taperw))
    CnrHatTable = CnBetaTable*lvt/b - CLTable*CLTable*0.4/(np.pi*ARw)
    CndATable = np.array([-0.0015, 0.0006, 0.0013, 0.0013, 0.0013, 0.0014, 0.0012, 0.0011,
                 0.0010, 0.0001, -0.0024, -0.0044, -0.0058, -0.0070, -0.0078, -0.0085,
                 -0.0091, -0.0095, -0.0097, -0.0098, -0.0099, -0.0099, -0.0099, -0.0099,
                 -0.0098, -0.0097, -0.0095, -0.0091, -0.0085, -0.0075, -0.0066, -0.0058,
                 -0.0050, -0.0042, -0.0034, -0.0025, -0.0017, -0.0009, 0])

    CndRTable = np.array([-0.0717, -0.0717, -0.0713, -0.0707, -0.0702, -0.0698, -0.0700, -0.0703,
                 -0.0706, -0.0704, -0.0699, -0.0689, -0.0676, -0.0662, -0.0648, -0.0635,
                 -0.0622, -0.0613, -0.0608, -0.0604, -0.0601, -0.0599, -0.0597, -0.0594,
                 -0.0589, -0.0581, -0.0568, -0.0546, -0.0510, -0.0451, -0.0398, -0.0348,
                 -0.0300, -0.0251, -0.0202, -0.0153, -0.0102, -0.0051, 0])
    CndASTable = -0.1*CndATable

    # Package outputs
    adb = ADB()
    adb.ACtype = ACtype
    adb.MODEL = MODEL
    adb.mSim = mSim
    adb.SMsim = SMsim
    adb.Ixx = Ixx
    adb.Iyy = Iyy
    adb.Izz = Izz
    adb.Ixz = Ixz
    adb.cBar = cBar
    adb.b = b
    adb.S = S
    adb.StaticThrust = StaticThrust
    adb.epsilon = epsilon
    adb.AlphaTable = AlphaTable
    adb.AlphaLength = AlphaLength
    adb.CLTable = CLTable
    adb.CLAlphaTable = CLAlphaTable
    adb.CLqHatTable = CLqHatTable
    adb.CLdETable = CLdETable
    adb.CLdFTable = CLdFTable
    adb.CLdSTable = CLdSTable
    adb.CDTable = CDTable
    adb.CDAlphaTable = CDAlphaTable
    adb.CDqHatTable = CDqHatTable
    adb.CDdETable = CDdETable
    adb.CDdFTable = CDdFTable
    adb.CDdSTable = CDdSTable
    adb.CmTable = CmTable
    adb.CmAlphaTable = CmAlphaTable
    adb.CmqHatTable = CmqHatTable
    adb.CmdETable = CmdETable
    adb.CmdFTable = CmdFTable
    adb.CmdSTable = CmdSTable
    adb.CYBetaTable = CYBetaTable
    adb.CYpHatTable = CYpHatTable
    adb.CYrHatTable = CYrHatTable
    adb.CYdATable = CYdATable
    adb.CYdRTable = CYdRTable
    adb.CYdASTable = CYdASTable
    adb.ClBetaTable = ClBetaTable
    adb.ClpHatTable = ClpHatTable
    adb.ClrHatTable = ClrHatTable
    adb.CldATable = CldATable
    adb.CldRTable = CldRTable
    adb.CldASTable = CldASTable
    adb.CnBetaTable = CnBetaTable
    adb.CnpHatTable = CnpHatTable
    adb.CnrHatTable = CnrHatTable
    adb.CndATable = CndATable
    adb.CndRTable = CndRTable
    adb.CndASTable = CndASTable
    return adb
