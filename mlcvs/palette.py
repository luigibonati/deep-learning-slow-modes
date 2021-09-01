###################################################################
# Fessa palette for python matplotlib
###################################################################
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb=tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return tuple([x/255. for x in rgb])


paletteFessa = [
          hex_to_rgb("#1F3B73"),
          hex_to_rgb("#2F9294"),
          hex_to_rgb("#50B28D"),
          hex_to_rgb("#A7D655"),
          hex_to_rgb("#FFE03E"),
          hex_to_rgb("#FFA955"),
          hex_to_rgb("#D6573B")
         ]

cm_fessa = LinearSegmentedColormap.from_list("fessa", paletteFessa, N=1000)

matplotlib.colors.ColorConverter.colors['fessa1'] = paletteFessa[0]
matplotlib.colors.ColorConverter.colors['fessa2'] = paletteFessa[1]
matplotlib.colors.ColorConverter.colors['fessa3'] = paletteFessa[2]
matplotlib.colors.ColorConverter.colors['fessa4'] = paletteFessa[3]
matplotlib.colors.ColorConverter.colors['fessa5'] = paletteFessa[4]
matplotlib.colors.ColorConverter.colors['fessa6'] = paletteFessa[5]
matplotlib.colors.ColorConverter.colors['fessa7'] = paletteFessa[6]

fessaNames=['fessa1', 'fessa2' , 'fessa3', 'fessa4', 
              'fessa5', 'fessa6' , 'fessa7']
