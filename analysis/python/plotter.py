import awkward as awk
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt, mpld3
import mplhep
import numpy
import uproot
import matplotlib.colors as colors

#import ROOT
#import root2matplot as r2mpl

import utils

#plt.rcParams.update({
#    "text.usetex",
#    "font.family",
#    "font.weight",
#    "font.size",
#})


def main() :
    
    l_filename = [
        "analysisOutputNtuple_17kelectrons.root:analysisOutputTree"
    ]
    
    h1_demo = hist.Hist(
        hist.axis.Regular(bins = 100, start = -30, stop = 30, name = "x")
    )
        
    eta_PCA_w_2D = hist.Hist(
        hist.axis.Regular(bins = 50, start = 1.4, stop = 3.2, name = "x", label='eta'),
        hist.axis.Regular(bins = 100, start = 0, stop = 100, name = "y", label='vertex difference (cm)'),
    )
    eta_PCA_unw_2D = hist.Hist(
        hist.axis.Regular(bins = 50, start = 1.4, stop = 3.2, name = "x", label='eta'),
        hist.axis.Regular(bins = 100, start = 0, stop = 100, name = "y"),
    )

    eta_polyfit_2D = hist.Hist(
        hist.axis.Regular(bins = 50, start = 1.4, stop = 3.2, name = "x", label='eta'),
        hist.axis.Regular(bins = 100, start = 0, stop = 100, name = "y"),
    )
    
    for tree_branches in uproot.iterate(
        files = l_filename,
        expressions = [
            "runNumber",
            "lumiNumber",
            "eventNumber",
            
            "ele_SC_energy",
            "ele_SC_eta",
            "ele_SC_phi",
            
            "ele_vtx_rho",
            "ele_vtx_z",
            
            # "ele_wpca_eigval0",
            # "ele_wpca_eigval1",
            
            # "ele_wpca_eigaxis0_p0",
            # "ele_wpca_eigaxis0_p1",
            
            # "ele_wpca_eigaxis1_p0",
            # "ele_wpca_eigaxis1_p1",


            "ele_vertex_distance_PCA_w",
            "ele_vertex_distance_PCA_unw", 
            "ele_vertex_distance_polyfit",
            "ele_vertex_distance_cyl_1.5cm",
            "ele_vertex_distance_cyl_1.75cm",
            "ele_vertex_distance_cyl_2cm",
            "ele_vertex_distance_cyl_2.5cm",
            "ele_vertex_distance_cyl_3cm",

            ##distance to fit
            "ele_distance_to_fit_PCA_w", 
            "ele_distance_to_fit_PCA_unw",
            "ele_distance_to_fit_polyfit",
            "ele_distance_to_fit_cyl_1.5cm",
            "ele_distance_to_fit_cyl_1.75cm",
            "ele_distance_to_fit_cyl_2cm",
            "ele_distance_to_fit_cyl_2.5cm",
            "ele_distance_to_fit_cyl_3cm",



            ##distance to fit
            "ele_chi_sq_PCA_w",
            "ele_chi_sq_PCA_unw",
            "ele_chi_sq_polyfit_w",
            "ele_chi_sq_cyl_1.5cm",
            "ele_chi_sq_cyl_1.75cm",
            "ele_chi_sq_cyl_2cm",
            "ele_chi_sq_cyl_2.5cm",
            "ele_chi_sq_cyl_3cm",
        ],
        language = utils.uproot_lang,
        num_workers = 10,
        #max_num_elements = 1,
        step_size = 100,
    ) :
        
        count = len(tree_branches["ele_vtx_z"])
        a_weights = numpy.ones(count) # dummy weights
        A=["ele_SC_energy", "ele_SC_eta", "ele_SC_ET"]

        B=["ele_vertex_distance_PCA_w",
            "ele_vertex_distance_PCA_unw", 
            "ele_vertex_distance_polyfit",
            "ele_vertex_distance_cyl_1.5cm",
            "ele_vertex_distance_cyl_1.75cm",
            "ele_vertex_distance_cyl_2cm",
            "ele_vertex_distance_cyl_2.5cm",
            "ele_vertex_distance_cyl_3cm",

            "ele_distance_to_fit_PCA_w", 
            "ele_distance_to_fit_PCA_unw",
            "ele_distance_to_fit_polyfit",
            "ele_distance_to_fit_cyl_1.5cm",
            "ele_distance_to_fit_cyl_1.75cm",
            "ele_distance_to_fit_cyl_2cm",
            "ele_distance_to_fit_cyl_2.5cm",
            "ele_distance_to_fit_cyl_3cm",


            "ele_chi_sq_PCA_w",
            "ele_chi_sq_PCA_unw",
            "ele_chi_sq_polyfit_w",
            "ele_chi_sq_cyl_1.5cm",
            "ele_chi_sq_cyl_1.75cm",
            "ele_chi_sq_cyl_2cm",
            "ele_chi_sq_cyl_2.5cm",
            "ele_chi_sq_cyl_3cm",
        ]


        bins_met=[160,160,160,160,160,160,160,160,50,50,50,50,50,50,50,50,100,100,100,100,100,100,100,100]
        start_met=[-80,-80,-80,-80,-80,-80,-80,-80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        stop_met=[80,80,80,80,80,80,80,80,50,50,50,50,50,50,50,50,1000,1000,1000,1000,1000,1000,1000,1000]
        #ll = [-80]*5 + [0]*10
        #bins=[eta, energy, energy transverse]
        bins_var=[50,20,20]
        start_var=[1.4,0,0]
        stop_var=[3.6,1000,100]
        r=0
        
        #fig = plt.figure(figsize = [10,5])
        for i in range(len(A)-1):
            for j in range(len(B)-1):
                hist_2D = hist.Hist(
                hist.axis.Regular(bins = bins_var[i], start= start_var[i], stop=stop_var[i], name = "x"),
                hist.axis.Regular(bins = bins_met[j], start = start_met[j], stop = stop_met[j], name = "y"),
                )
            
                hist_2D.fill(
                    tree_branches[A[i]],
                    tree_branches[B[j]],
                    weight = a_weights
                ) 
                #r+=1
                fig = plt.figure(figsize = [10,5])
                ax = fig.add_subplot(1,1,1)
                ax.set_title(f"{A[i]} vs {B[j]}")
                mplhep.hist2dplot(hist_2D, ax = ax, cbar=False)
                mpl.show()
                ##now add profile graph
                #  h1_demo = hist.Hist(
                #       hist.axis.Regular(bins = 100, start = -30, stop = 30, name = "x")
                # )
                #profx = hist_2D.profile("x")
                #mplhep.histplot(profx, ax = ax1)





    
    
    
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.set_title('PCA weighted')
    # ax2.set_title('PCA unweighted')
    # ax3.set_title('Polyfit weighted')
    
    #
    #fig2 = plt.figure(figsize = [10, 8])
    #ax2 = fig2.add_subplot(1, 1, 1)
    
  
    
    
    fig.canvas.draw()
    fig.show()
    fig.canvas.flush_events()
    mpld3.save_html(fig, "/Users/EmilyRichards/Desktop/plot_demo.html")


    print("hi")

    
    return 0


if (__name__ == "__main__") :
    
    main()