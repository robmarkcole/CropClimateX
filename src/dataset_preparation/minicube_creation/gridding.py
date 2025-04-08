import numpy as np
from shapely.ops import split
import geopandas
from shapely.geometry import MultiPolygon, Polygon, LineString, box
from itertools import product

def create_patch_mask(patches, bounds):
    patch_mask = np.zeros(bounds, dtype=np.int8)
    for rect in patches:
        patch_mask[rect[0]:rect[2], rect[1]:rect[3]] += 1
    return patch_mask

def get_squares_from_rect(RectangularPolygon, side_length):
    """
    Divide a Rectangle (Shapely Polygon) into squares of equal area.

    `side_length` : required side of square

    """
    rect_coords = np.array(RectangularPolygon.boundary.coords.xy)
    y_list = rect_coords[1]
    x_list = rect_coords[0]
    y1 = min(y_list)
    y2 = max(y_list)
    x1 = min(x_list)
    x2 = max(x_list)
    width = x2 - x1
    height = y2 - y1

    xcells = int(np.round(width / side_length))
    ycells = int(np.round(height / side_length))

    yindices = np.linspace(y1, y2, ycells + 1)
    xindices = np.linspace(x1, x2, xcells + 1)
    horizontal_splitters = [
        LineString([(x, yindices[0]), (x, yindices[-1])]) for x in xindices
    ]
    vertical_splitters = [
        LineString([(xindices[0], y), (xindices[-1], y)]) for y in yindices
    ]
    result = RectangularPolygon
    for splitter in vertical_splitters:
        result = MultiPolygon(split(result, splitter))
    for splitter in horizontal_splitters:
        result = MultiPolygon(split(result, splitter))
    square_polygons = list(result.geoms)

    return square_polygons


def split_polygon_evenly(shp, side_length, thresh=0.9):
    """
    Overlay fishnet over shp structure with same shaped polygons with approx. length of side_length.
    Removes non-intersecting polygons. 
    code from here: https://stackoverflow.com/questions/8491927/algorithm-to-subdivide-a-polygon-in-smaller-polygons

    """
    assert side_length>0, "side_length must be a float>0"
    Rectangle    = shp.envelope
    squares      = get_squares_from_rect(Rectangle, side_length=side_length)
    SquareGeoDF  = geopandas.GeoDataFrame(squares).rename(columns={0: "geometry"})
    SquareGeoDF.set_geometry('geometry', inplace=True)
    Geoms        = SquareGeoDF[SquareGeoDF.intersects(shp)].geometry.values
    geoms = [g for g in Geoms if ((g.intersection(shp)).area / g.area) >= thresh]
    return geoms

def get_squares_from_mask(mask, side_length, start_point, ignore_mask):
    if ignore_mask:
        minx, maxx = 10, mask.shape[0]
        miny, maxy = 0, mask.shape[1]
    else:
        minx, maxx = np.min(np.argwhere(mask == 1)[:,0]), np.max(np.argwhere(mask == 1)[:,0])
        miny, maxy = np.min(np.argwhere(mask == 1)[:,1]), np.max(np.argwhere(mask == 1)[:,1])
    
    if start_point == "top_left":
        xindices = np.arange(minx, maxx+side_length, side_length)
        yindices = np.arange(miny, maxy+side_length, side_length)
    elif start_point == "bottom_left":
        xindices = np.arange(maxx-side_length, minx-side_length, -side_length)
        yindices = np.arange(miny, maxy+side_length, side_length)
    elif start_point == "top_right":
        xindices = np.arange(minx, maxx+side_length, side_length)
        yindices = np.arange(maxy-side_length, miny-side_length, -side_length)
    elif start_point == "bottom_right":
        xindices = np.arange(maxx-side_length, minx-side_length, -side_length)
        yindices = np.arange(maxy-side_length, miny-side_length, -side_length)
    else:
        raise ValueError("Invalid start_point. Choose from 'top_left', 'top_right', 'bottom_left', 'bottom_right'.")
    grid = product(xindices, yindices)
    squares = [(x, y, x+side_length, y+side_length) for x,y in grid]

    return squares

def split_image_mask_to_polygons(mask, side_length, thresh=0.9, start_point="bottom_left", ignore_mask=False):
    """
    overlay grid of length side_length over a polygon defined by mask. start to create grid at start_point.
    """
    assert side_length>0, "side_length must be a float>0"
    squares = get_squares_from_mask(mask, side_length=side_length, start_point=start_point, ignore_mask=ignore_mask)
    patch_masks = [create_patch_mask([square], mask.shape) for square in squares]
    squares = [s for p,s in zip(patch_masks, squares) if (np.sum(p[mask]) / side_length**2) >= thresh]
    return squares

def get_squares_from_shp(shp, side_length, start_point):
    rect_coords = np.array(shp.boundary.coords.xy)
    x_list = rect_coords[0]
    y_list = rect_coords[1]
    minx = min(x_list)
    maxx = max(x_list)
    miny = min(y_list)
    maxy = max(y_list)

    if start_point == "bottom_left":
        xindices = np.arange(minx, maxx+side_length, side_length)
        yindices = np.arange(miny, maxy+side_length, side_length)
    elif start_point == "bottom_right":
        xindices = np.arange(maxx-side_length, minx-side_length, -side_length)
        yindices = np.arange(miny, maxy+side_length, side_length)
    elif start_point == "top_left":
        xindices = np.arange(minx, maxx+side_length, side_length)
        yindices = np.arange(maxy-side_length, miny-side_length, -side_length)
    elif start_point == "top_right":
        xindices = np.arange(maxx-side_length, minx-side_length, -side_length)
        yindices = np.arange(maxy-side_length, miny-side_length, -side_length)
    else:
        raise ValueError("Invalid start_point. Choose from 'top_left', 'top_right', 'bottom_left', 'bottom_right'.")

    grid = product(xindices, yindices)
    squares = [(x, y, x+side_length, y+side_length) for x,y in grid]
    squares = [Polygon(box(*square)) for square in squares]

    return squares

def split_shp_to_polygons(shp, side_length, thresh=0.9, start_point="bottom_left"):
    """
    overlay grid of length side_length over a polygon shp. start to create grid at start_point.
    """
    assert side_length>0, "side_length must be a float>0"
    Rectangle    = shp.envelope
    squares      = get_squares_from_shp(Rectangle, side_length=side_length, start_point=start_point)
    SquareGeoDF  = geopandas.GeoDataFrame(squares).rename(columns={0: "geometry"})
    SquareGeoDF.set_geometry('geometry', inplace=True)
    Geoms        = SquareGeoDF[SquareGeoDF.intersects(shp)].geometry.values
    geoms = [g for g in Geoms if ((g.intersection(shp)).area / g.area) >= thresh]
    return geoms

def pixels_idx_to_coord(idxs, ref_raster):
    """take pixel indices and turn them into coordinates according to xarray ref_raster
        idxs take the form: [(minx, miny, maxx, maxy), ...]
    """
    def _get_coordinate(idx, coords, length, resolution, neg_direction):
        if idx >= length:
            org = coords[length-1].values.item()
            add = resolution * (idx - length + 1)
            if neg_direction: # correct direction for negative signed coords
                add *= -1
            return org + add
        elif idx < 0:
            org = coords[0].values.item()
            add = resolution * idx
            if not neg_direction: # correct direction for negative signed coords
                add *= -1
            return org + add
        else:
            return coords[idx].values.item()
        
    out = []
    for min_y, min_x, max_y, max_x in idxs:
        min_coord_x = _get_coordinate(min_x, ref_raster.coords['x'], len(ref_raster.coords['x']), ref_raster.rio.resolution()[0],neg_direction=True)
        min_coord_y = _get_coordinate(min_y, ref_raster.coords['y'], len(ref_raster.coords['y']), ref_raster.rio.resolution()[0],neg_direction=False)
        max_coord_x = _get_coordinate(max_x, ref_raster.coords['x'], len(ref_raster.coords['x']), ref_raster.rio.resolution()[0],neg_direction=False)
        max_coord_y = _get_coordinate(max_y, ref_raster.coords['y'], len(ref_raster.coords['y']), ref_raster.rio.resolution()[0],neg_direction=True)
        out.append((min_coord_x, min_coord_y, max_coord_x, max_coord_y))
    return out

def coord_to_pixels_idx(coords, ref_raster):
    """take coordinates and turn them into pixel indices according to xarray ref_raster
        coords take the form: [(minx, miny, maxx, maxy), ...]
    """
    return [(ref_raster.indexes['x'].get_indexer([minx], method='nearest')[0],
             ref_raster.indexes['y'].get_indexer([miny], method='nearest')[0],
             ref_raster.indexes['x'].get_indexer([maxx], method='nearest')[0],
             ref_raster.indexes['y'].get_indexer([maxy], method='nearest')[0])
               for minx, miny, maxx, maxy in coords]