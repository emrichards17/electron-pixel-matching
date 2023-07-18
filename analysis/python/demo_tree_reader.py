import awkward
import uproot


import utils


def main() :
    
    l_filename = [
        "ntupleTree.root:treeMaker/tree",
    ]
    
    for tree_branches in uproot.iterate(
        files = l_filename,
        expressions = [
            "genEle_count",
            "v_genEle_pt",
            
            "hgcalEle_count",
            "v_hgcalEle_pt",
            
            "v_hgcalEle_matchedGenEle_idx",
            
            "v_hgcalEle_SC_clus_count",
            "vv_hgcalEle_SC_clus_energy",
            
            "v_hgcalEle_SC_hit_count",
            "vv_hgcalEle_SC_hit_energy",
        ],
        cut = "(hgcalEle_count > 0) & (count_nonzero(v_hgcalEle_matchedGenEle_idx > 0, axis = -1) > 0)",
        language = utils.uproot_lang,
        num_workers = 10,
        #entry_start = 0,
        #entry_stop = 10
    ) :
        
        #print(tree_branches["hgcalEle_count"])
        print(type(tree_branches))
        
        print(tree_branches["v_hgcalEle_pt"])
        print(tree_branches["v_hgcalEle_SC_clus_count"])
        print(tree_branches["vv_hgcalEle_SC_clus_energy"])
        print(tree_branches["v_hgcalEle_SC_hit_count"])
        print(tree_branches["vv_hgcalEle_SC_hit_energy"])
    
    return 0


if (__name__ == "__main__") :
    
    main()