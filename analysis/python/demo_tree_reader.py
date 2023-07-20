import awkward
import uproot


import utils


def main() :
    
    l_filename = [
        "ntupleTree.root:treeMaker/tree",
    ]
    
    print("")
    
    for tree_branches in uproot.iterate(
        files = l_filename,
        expressions = [
            #"genEle_count",
            #"v_genEle_pt",
            
            "hgcalEle_count",
            "v_hgcalEle_charge",
            "v_hgcalEle_energy",
            "v_hgcalEle_pt",
            "v_hgcalEle_eta",
            "v_hgcalEle_phi",
            
            "v_hgcalEle_matchedGenEle_idx",
            
            #"v_hgcalEle_SC_clus_count",
            #"vv_hgcalEle_SC_clus_energy",
            
            "v_hgcalEle_SC_hit_count",
            "vv_hgcalEle_SC_hit_energyTrue",
            #"vv_hgcalEle_SC_hit_energy",
            #"vv_hgcalEle_SC_hit_energyFrac",
            "vv_hgcalEle_SC_hit_rho",
            "vv_hgcalEle_SC_hit_x",
            "vv_hgcalEle_SC_hit_y",
            "vv_hgcalEle_SC_hit_z",
            "vv_hgcalEle_SC_hit_eta",
            "vv_hgcalEle_SC_hit_phi",
            "vv_hgcalEle_SC_hit_detector",
            "vv_hgcalEle_SC_hit_layer",
            
            "pixelRecHit_count",
            "v_pixelRecHit_globalPos_rho",
            "v_pixelRecHit_globalPos_x",
            "v_pixelRecHit_globalPos_y",
            "v_pixelRecHit_globalPos_z",
        ],
        aliases = {
            "vv_hgcalEle_SC_hit_energyTrue": "vv_hgcalEle_SC_hit_energy*vv_hgcalEle_SC_hit_energyFrac"
        },
        cut = "(hgcalEle_count > 0)",
        language = utils.uproot_lang,
        num_workers = 10,
        #max_num_elements = 10,
        step_size = 100,
    ) :
        
        #print(tree_branches["hgcalEle_count"])
        print(type(tree_branches))
        
        hgcalEle_SC_hits = awkward.zip({
            "energy": tree_branches["vv_hgcalEle_SC_hit_energyTrue"],
            "rho": tree_branches["vv_hgcalEle_SC_hit_rho"],
            "x": tree_branches["vv_hgcalEle_SC_hit_x"],
            "y": tree_branches["vv_hgcalEle_SC_hit_y"],
            "z": tree_branches["vv_hgcalEle_SC_hit_z"],
            "eta": tree_branches["vv_hgcalEle_SC_hit_eta"],
            "phi": tree_branches["vv_hgcalEle_SC_hit_phi"],
            "detector": tree_branches["vv_hgcalEle_SC_hit_detector"],
            "layer": tree_branches["vv_hgcalEle_SC_hit_layer"],
        })
        
        hgcalEles = awkward.zip({
            "charge": tree_branches["v_hgcalEle_charge"],
            "pt": tree_branches["v_hgcalEle_pt"],
            "eta": tree_branches["v_hgcalEle_eta"],
            "phi": tree_branches["v_hgcalEle_phi"],
            "energy": tree_branches["v_hgcalEle_energy"],
            "SC_hit_count": tree_branches["v_hgcalEle_SC_hit_count"],
            "SC_hits": hgcalEle_SC_hits,
        })
        
        pixelRecHits = awkward.zip({
            "rho": tree_branches["v_pixelRecHit_globalPos_rho"],
            "x": tree_branches["v_pixelRecHit_globalPos_x"],
            "y": tree_branches["v_pixelRecHit_globalPos_y"],
            "z": tree_branches["v_pixelRecHit_globalPos_z"],
        })
        
        print("")
    
    return 0


if (__name__ == "__main__") :
    
    main()