category = {
    'Apple': (
        0 , {
        'Apple_scab': 0,
        'Black_rot': 1,
        'Cedar_apple_rust': 2,
        'healthy': 3,
        }
    ),
    'Blueberry': (
        1, {
        'healthy': 4,
        }
    ),
    'Cherry': (
        2, {
            'Powdery_mildew': 5,
            'healthy': 6,
        }
    ),
    'Corn': (
        3, {
            'Cercospora_leaf_spot': 7,
            'Common_rust': 8,
            'Northern_Leaf_Blight': 9,
            'healthy': 10,
        }
    ),
    'Grape': (
        4, {
            'Black_rot': 11,
            'Esca': 12,
            'Leaf_blight': 13,
            'healthy': 14,
        }
    ),
    'Orange': (
        5, {
            'Haunglongbing': 15,
        }
    ),
    'Peach': (
        6, {
            'Bacterial_spot': 16,
            'healthy': 17,
        }
    ),
    'Pepper': (
        7, {
            'Bacterial_spot': 18,
            'healthy': 19
        }
    ),
    'Potato': (
        8, {
            'Early_blight': 20,
            'Late_blight': 21,
            'healthy': 22,
        }
    ),
    'Raspberry': (
        9, {
            'healthy': 23,
        }
    ),
    'Soybean': (
        10, {
            'healthy': 24,
        }
    ),
    'Squash': (
        11, {
            'Powdery_mildew': 25,
        }
    ),
    'Strawberry': (
        12, {
            'Leaf_scorch': 26,
            'healthy': 27,
        }
    ),
    'Tomato': (
        13, {
            'Bacterial_spot': 28,
            'Early_blight': 29,
            'Late_blight': 30,
            'Leaf_Mold': 31,
            'Septoria_leaf_spot': 32,
            'Spider_mites': 33,
            'Target_Spot': 34,
            'Yellow_Leaf_Curl_Virus': 35,
            'mosaic_virus': 36,
            'healthy': 37,
        }
    ),
}


def idx2category(idx, fruit):
    if not (0 <= idx < 37):
        return None
    
    for disease, idx_here in category[fruit][1].items():
        if idx_here == idx:
            return disease
        