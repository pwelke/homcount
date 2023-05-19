from ghc.generate_k_tree import partial_ktree_sample, Nk_strategy, product_graph_ktree_profile, min_kernel, full_kernel, random_ktree_profile, large_pattern, min_kernel, full_kernel
from ghc.utils.fast_weisfeiler_lehman import wl_kernel

def get_hom_profile(f_str):
    if f_str == "random_ktree":
        return random_ktree_profile
    elif f_str == "min_kernel":
        return min_kernel
    elif f_str == "full_kernel":
        return full_kernel
    elif f_str == "wl_kernel":
        return wl_kernel
    elif f_str == 'large_pattern':
        return large_pattern
    elif f_str == 'product_graph_ktree_profile':
        return product_graph_ktree_profile        
    else:  # Return all posible options
        return ["random_ktree", 
                "product_graph_ktree_profile", "min_kernel", 
                "full_kernel", 'wl_kernel', 'large_pattern']
