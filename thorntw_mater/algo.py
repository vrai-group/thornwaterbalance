"""The Thornthwaite-Mather water balance (Thornthwaite & Mather, 1955; Thornthwaite & Mather, 1957) uses an accounting procedure 
to analyze the allocation of water among various components of the hydrologic system."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

URL_K_LAT_DATA = "https://thornwaterbalanceapp.s3.amazonaws.com/data/K_latN_S.csv"

def thorntw_mater_proc(source_path, LAT, SM, SRT, beta, mean_calc, file_out, img_out, plot_show = False):
    """
    This function is a python implementation of the thornwater balance
    Parameters
    ----------
    source_path : string
        path for input user data file
    LAT : int
        latitude of the area of interest
    SM: float
        SM = soil moisture storage capacity value (mm)
    SRT: float
        snowfall rainfall threshold (°C)
    beta: float
        dimensionless runoff coefficient (%)
    mean_calc: bool
        activate the calculation taking the average for each month.       
    file_out : string
        path to save output
    img_out : string
        path to save plots as figures
    Returns
    -------
    """
    try:
        # loading the lat template data
        k = np.genfromtxt(URL_K_LAT_DATA, delimiter=';', filling_values=0)

        # user data loading
        training_data_x = pd.read_excel(source_path)
        values = training_data_x.values
        data = np.array(values)

        # management of MEAN_CALC MODE
        proc = True
        global_count = 0

        if mean_calc:
            # we calculate average for each month
            mean_values = np.zeros((12, 4))
            for i in range(1, 13):
                print(i)
                mean_values[i-1, :] = np.mean(training_data_x.loc[training_data_x.iloc[:, 1] == i])
            data = mean_values

        while proc:

            if global_count >= 2:
                break
            global_count += 1

            k = k[:, k[0] == LAT]
            Tm = data[:, 2]
            im = (Tm/5) ** 1.514
            k1 = np.unique(data[:, 0])
            k1d = {}

            for j in range(len(k1)):
                index = [data[:, 0] == k1[j]]
                Itmp = im[index]
                k1d[int(k1[j])] = np.nansum(Itmp)

            II = np.zeros((len(data)))

            for j in range(len(data)):
                II[j] = float(k1d[int(data[j, 0])])

            n = len(data)
            npp = np.resize(k[1:], n)

            II = np.array(II)
            npp = np.array(npp)
            Tm = np.array(Tm)

            a = (675 * 10**-9 * (II**3)) - (771 * 10**-7 *
                                            (II**2)) + (1792 * 10**-5 * II) + 0.49239
            a = np.round(a, 2)

            PET = 16 * npp * (((10 * Tm) / II) ** a)
            PET = np.array(PET)
            PET = np.round(PET, 1)
            where_are_NaNs = np.isnan(PET)
            PET[where_are_NaNs] = 0

            threshold = 0
            substitute = 0
            PET[PET <= threshold] = substitute

            P = data[:, 3]
            delta = P - PET
            delta = np.array(delta)

            P1 = [None] * len(P)
            for ih in range(len(P)):
                P1[ih] = P[ih]
            P1 = np.array(P1)
            P1[Tm > SRT] = 0
            SP = np.convolve(P1, np.ones(10, dtype=float), 'full')
            SP = SP[0:len(P1)]
            SP1 = np.zeros((len(SP),))

            for d in range(len(SP)):
                if d == 0:
                    SP1[d] = SP[d]
                else:
                    if SP[d - 1] < SP[d]:
                        SP1[d - 1] = 0
                        SP1[d] = SP[d]
                    else:
                        SP1[d] = SP[d]

            for di in range(len(SP1)):
                if SP1[di] != 0:
                    if di == 0:
                        SP1[di] = SP1[di]
                    else:
                        if SP1[di] < SP1[di-1]:
                            SP1[di] = 0
                        else:
                            SP1[di] = SP1[di]
                else:
                    SP1[di] = SP1[di]

            SMRO = np.zeros((len(SP1) + 1,))
            ii = 0
            for h in range(len(SP1)):
                if SP1[h] > 0:
                    if ii == 0:
                        SMRO[h] = 0
                    ii = ii+1
                    if ii == 1:
                        SMRO[h + 1] = SP1[h]*0.10
                    elif ii == 2:
                        SMRO[h + 1] = (SP1[h]-SMRO[h - 1])/2
                    else:
                        SMRO[h + 1] = SMRO[h]/2
                else:
                    ii = 0
                    SMRO[h + 1] = SP1[h]

            SMRO = SMRO[0:len(P1)]
            SMRO = np.round(SMRO, 1)
            S = np.zeros((len(SP),))
            S[0] = P[0] - PET[0]
            AET = np.zeros((len(SP),))
            ST = np.zeros((len(SP),))

            ST.astype(float)
            AET.astype(float)
            PET.astype(float)
            P.astype(float)

            for h in range(len(Tm)):
                if Tm[h] >= SRT:
                    if delta[h] < 0:
                        S[h] = 0
                        if h == 0:
                            ST[h] = SM - (SM * (1 - np.exp(-(PET[h] - P[h])/SM)))
                            AET[h] = PET[h]
                        else:
                            ST[h] = ST[h - 1] - \
                                (ST[h - 1] * (1 - np.exp(-(PET[h] - P[h])/SM)))
                            AET[h] = P[h] + \
                                (ST[h - 1] * (1 - np.exp(-(PET[h] - P[h])/SM)))
                    else:
                        if h == 0:
                            ST[h] = SM
                            AET[h] = PET[h]
                        else:
                            if delta[h] < (SM - ST[h - 1]):
                                ST[h] = ST[h - 1] + delta[h]
                                S[h] = 0
                                AET[h] = PET[h]
                            else:
                                ST[h] = float(SM)
                                AET[h] = PET[h]
                                if ST[h - 1] > SM:
                                    S[h] = delta[h]
                                else:
                                    S[h] = delta[h]-(SM-ST[h-1])
                else:
                    if h == 0:
                        ST[h] = SM
                        AET[h] = 0
                        S[h] = 0
                    else:
                        ST[h] = ST[h-1] + delta[h]
                        AET[h] = 0
                        S[h] = 0

            RO = np.zeros((len(SP),))
            RES = np.zeros((len(SP),))

            for h in range(len(S)):
                if h == 0:
                    RO[h] = beta * (0 + S[h])
                    RES[h] = (1 - beta)*(0 + S[h])
                elif h == 1:
                    RO[h] = (S[h] + RES[h-1]) * beta
                    RES[h] = (1 - beta) * (RES[h-1] + S[h])
                else:
                    RO[h] = (RES[h-1] + S[h]) * beta
                    RES[h] = (1 - beta) * (RES[h-1] + S[h])

            RO = np.round(RO, 1)
            tot_RO = np.zeros((len(SP),))
            for f in range(len(S)):
                tot_RO[f] = SMRO[f]+RO[f]

            out1 = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3], npp, II, a, PET, delta, AET, ST, S, RO, RES, SMRO, tot_RO]).transpose()

            if mean_calc:
                data = out1[:, 0:4]
            else:
                break

        # results exporting
        col = ['year', 'month', 'Tm', 'P', 'k', 'I', 'a', 'PET', 'delta', 'AET', 'ST', 'S', 'RO', 'RES', 'SMRO', 'TOT_RO']
        if mean_calc:
            col = ['month', 'Tm', 'P', 'k', 'I', 'a', 'PET', 'delta', 'AET', 'ST', 'S', 'RO', 'RES', 'SMRO', 'TOT_RO']
            out1 = out1[:, 1:]

        # saving as Exlcel xlsx format
        df = pd.DataFrame(out1, columns=col)
        df.to_excel(file_out, sheet_name='results')

        # plot data in subplots
        precip = data[:, 3]
        datatime = np.arange(0, len(Tm), 1)
        fig, axs = plt.subplots(5, 2)
        axs[0, 0].plot(np.float64(datatime), np.float64(Tm), '--', color='black')
        axs[0, 0].set_ylim([min(Tm)-10, max(Tm)+10])
        axs[0, 0].set_ylabel('Temp. °C')

        axs[0, 1].bar(np.float64(datatime), np.float64(precip), color='black')
        axs[0, 1].set_ylim([min(precip), max(precip)+20])
        axs[0, 1].set_ylabel('Rainfall [mm]')

        axs[1, 0].plot(np.float64(datatime), np.float64(PET), color='magenta')
        axs[1, 0].set_ylim([min(PET), max(PET)+10])
        axs[1, 0].set_ylabel('PET [mm]')

        axs[1, 1].bar(np.float64(datatime), np.float64(delta), color='cyan')
        axs[1, 1].set_ylim([min(delta), max(delta)+10])
        axs[1, 1].set_ylabel('P-PET [mm]')

        axs[2, 0].plot(np.float64(datatime), np.float64(AET), color='orange')
        axs[2, 0].set_ylim([min(AET), max(AET)+10])
        axs[2, 0].set_ylabel('AET [mm]')

        axs[2, 1].plot(np.float64(datatime), np.float64(ST), color='red')
        axs[2, 1].set_ylim([min(ST), max(ST)+10])
        axs[2, 1].set_ylabel('ST [mm]')

        axs[3, 0].plot(np.float64(datatime), np.float64(S), color='yellow')
        axs[3, 0].set_ylim([min(S), max(S)])
        axs[3, 0].set_ylabel('S [mm]')

        axs[3, 1].plot(np.float64(datatime), np.float64(RO), color='green')
        axs[3, 1].set_ylim([min(RO), max(RO)])
        axs[3, 1].set_ylabel('RO [mm]')

        axs[4, 0].plot(np.float64(datatime), np.float64(SMRO), color='blue')
        axs[4, 0].set_ylim([min(SMRO), max(SMRO)])
        axs[4, 0].set_ylabel('SMRO [mm]')

        axs[4, 1].plot(np.float64(datatime), np.float64(tot_RO), color='black')
        axs[4, 1].set_ylim([min(tot_RO), max(tot_RO)])
        axs[4, 1].set_ylabel('tot RO [mm]')

        for ax in axs.flat:
            ax.set(xlabel='Time [months]')
            ax.set_xticks(np.arange(min(datatime), max(datatime)+1, 12))
            ax.xaxis.label.set_size(24)
            ax.yaxis.label.set_size(24)
            ax.set_xticklabels(np.arange(min(datatime), max(datatime)+1, 12), rotation=0, fontsize=24)
            ax.tick_params(labelsize=24)

        plt.rcParams.update({'font.size': 26})

        figure = plt.gcf()
        figure.set_size_inches(32, 30)

        plt.savefig(img_out, bbox_inches='tight', dpi=300)
        if plot_show:
            plt.show()
    except Exception as e:
        print("Error:" + str(e))
        return -1
    return 0