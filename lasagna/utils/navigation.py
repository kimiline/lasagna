"""
Functions that move the views to a point and pick the colour that marks selected points
"""
import numpy as np


def highlight_color(color):
    """
    The colour that marks selected points drawn in color: white, or magenta if color is near-white
    """
    return [255, 0, 255] if min(color) > 200 else [255, 255, 255]


def jump_to_point(lasagna, point):
    """
    Show the slices through point ([Z, X, Y] in stack voxels) in all three views of lasagna, and
    pan any view where the point is out of sight. Only redraws until an image stack is loaded.
    """
    stacks = lasagna.returnIngredientByType("imagestack")
    if not stacks:
        lasagna.update_2D_plot_ingredients_in_axes()
        return

    point = [int(round(value)) for value in point]
    for axis in lasagna.axes2D:
        n_slices = max(stack.data(axis.axisToPlot).shape[0] for stack in stacks)
        slice_to_plot = min(max(point[axis.axisToPlot], 0), n_slices - 1)
        axis.updatePlotItems_2D(lasagna.ingredientList, sliceToPlot=slice_to_plot)

        # The point in this view's coordinates, as sparsepoints.data places it
        x, y = np.delete(point, axis.axisToPlot)
        if axis.axisToPlot == 2:
            x, y = y, x
        view_box = axis.view.getViewBox()
        (x0, x1), (y0, y1) = view_box.viewRange()
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            half_w, half_h = (x1 - x0) / 2, (y1 - y0) / 2
            view_box.setRange(xRange=(x - half_w, x + half_w), yRange=(y - half_h, y + half_h), padding=0)
