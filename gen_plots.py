'''
auxiliary funcs to generate latex-friendly plots
based on: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def set_size(width='thesis', fraction=1, subplot=[1, 1]):
    '''
    Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    '''
    if width == 'thesis':
        width_pt = 426.79135
    else:
        width_pt = width

    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


if __name__ == '__main__':
	plt.style.use('seaborn')
	width = 345

	nice_fonts = {
	        # Use LaTeX to write all text
	        'text.usetex': True,
	        'font.family': 'serif',
	        # Use 10pt font in plots, to match 10pt font in document
	        'axes.labelsize': 10,
	        'font.size': 10,
	        # Make the legend/label fonts a little smaller
	        'legend.fontsize': 8,
	        'xtick.labelsize': 8,
	        'ytick.labelsize': 8,
	}

	mpl.rcParams.update(nice_fonts)

	x = np.linspace(0, 2*np.pi, 100)
	fig, ax = plt.subplots(1, 1, figsize=set_size(width))
	ax.plot(x, np.sin(x))
	ax.set_xlim(0, 2*np.pi)
	ax.set_xlabel(r'$\theta$')
	ax.set_ylabel(r'$\sin{(\theta)}$')

	plt.savefig('/home/ana/test.svg', format='svg', bbox_inches='tight')