import matplotlib.collections as mcollections
import matplotlib.lines as mlines

import matplotlib.collections as mcollections
import matplotlib.lines as mlines

def set_selective_rasterization(ax, 
                                rasterize_markers=None, 
                                rasterize_collections=None,
                                rasterize_linestyles=None,
                                raster_zorder=-1,
                                threshold=0,
                                preserve_other_zorder=True):
    """
    Selectively rasterize plot elements in a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to modify
    rasterize_markers : list of str, optional
        List of marker types to rasterize (e.g., ['.', 'o', '*'])
        If None, marker type is not used as a criterion
    rasterize_collections : list of type, optional
        List of collection types to rasterize 
        (e.g., [mcollections.PathCollection, mcollections.LineCollection])
        If None, collection type is not used as a criterion
    rasterize_linestyles : list of str, optional
        List of line styles to rasterize (e.g., ['-', '--'])
        If None, linestyle is not used as a criterion
    raster_zorder : float, default=-1
        zorder value for rasterized elements (must be < threshold)
    threshold : float, default=0
        Rasterization threshold. Elements with zorder < threshold are rasterized
    preserve_other_zorder : bool, default=True
        If True, only modify zorder of elements being rasterized.
        If False, set non-rasterized elements to their original zorder (no change).
    
    Notes
    -----
    When both rasterize_markers and rasterize_linestyles are specified,
    a line will be rasterized if it matches EITHER criterion (OR logic).
    Elements not selected for rasterization are left unchanged by default.
    """
    
    # Handle collections
    for collection in ax.collections:
        should_rasterize = False
        
        if rasterize_collections is not None:
            # Check if collection type is in the rasterize list
            should_rasterize = any(isinstance(collection, ctype) 
                                  for ctype in rasterize_collections)
        
        if should_rasterize:
            collection.set_zorder(raster_zorder)
        # If preserve_other_zorder is True, don't touch non-rasterized elements
    
    # Handle line plots
    for line in ax.lines:
        should_rasterize = False
        
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        
        # Check marker criteria
        marker_match = False
        if rasterize_markers is not None:
            if marker != 'None' and marker in rasterize_markers:
                marker_match = True
        
        # Check linestyle criteria  
        linestyle_match = False
        if rasterize_linestyles is not None:
            if linestyle != 'None' and linestyle in rasterize_linestyles:
                linestyle_match = True
        
        # OR logic: rasterize if EITHER condition is met
        if rasterize_markers is not None or rasterize_linestyles is not None:
            should_rasterize = marker_match or linestyle_match
        
        if should_rasterize:
            line.set_zorder(raster_zorder)
        # If preserve_other_zorder is True, don't touch non-rasterized elements
    
    # Set the rasterization threshold
    ax.set_rasterization_zorder(threshold)

def diagnose_axis_elements(ax, verbose=True):
    """
    Diagnose all plot elements in an axis that could be selectively rasterized.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to diagnose
    verbose : bool, default=True
        If True, print detailed information. If False, return data structure only.
    
    Returns
    -------
    dict
        Dictionary containing information about all rasterizable elements:
        {
            'collections': list of dicts with collection info,
            'lines': list of dicts with line info,
            'summary': dict with counts by type
        }
    """
    
    diagnosis = {
        'collections': [],
        'lines': [],
        'summary': {
            'total_collections': 0,
            'total_lines': 0,
            'collection_types': {},
            'marker_types': {},
            'linestyle_types': {}
        }
    }
    
    # Analyze collections
    if verbose:
        print("=" * 70)
        print("COLLECTIONS (scatter, fill_between, etc.)")
        print("=" * 70)
    
    for i, collection in enumerate(ax.collections):
        ctype = type(collection)
        ctype_name = ctype.__name__
        
        # Get collection properties
        info = {
            'index': i,
            'type': ctype,
            'type_name': ctype_name,
            'zorder': collection.get_zorder(),
            'label': collection.get_label() if hasattr(collection, 'get_label') else None
        }
        
        # Try to get additional properties
        try:
            info['facecolor'] = collection.get_facecolor()
        except:
            pass
        
        try:
            info['edgecolor'] = collection.get_edgecolor()
        except:
            pass
        
        diagnosis['collections'].append(info)
        
        # Update summary
        diagnosis['summary']['total_collections'] += 1
        diagnosis['summary']['collection_types'][ctype_name] = \
            diagnosis['summary']['collection_types'].get(ctype_name, 0) + 1
        
        if verbose:
            print(f"\nCollection {i}:")
            print(f"  Type: {ctype_name}")
            print(f"  Full type: {ctype}")
            print(f"  Label: {info['label']}")
            print(f"  Current zorder: {info['zorder']}")
    
    # Analyze lines
    if verbose:
        print("\n" + "=" * 70)
        print("LINES (plot, with markers and linestyles)")
        print("=" * 70)
    
    for i, line in enumerate(ax.lines):
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        
        info = {
            'index': i,
            'marker': marker,
            'linestyle': linestyle,
            'color': line.get_color(),
            'linewidth': line.get_linewidth(),
            'label': line.get_label(),
            'zorder': line.get_zorder()
        }
        
        diagnosis['lines'].append(info)
        
        # Update summary
        diagnosis['summary']['total_lines'] += 1
        
        if marker != 'None':
            diagnosis['summary']['marker_types'][marker] = \
                diagnosis['summary']['marker_types'].get(marker, 0) + 1
        
        if linestyle != 'None':
            diagnosis['summary']['linestyle_types'][linestyle] = \
                diagnosis['summary']['linestyle_types'].get(linestyle, 0) + 1
        
        if verbose:
            print(f"\nLine {i}:")
            print(f"  Label: {info['label']}")
            print(f"  Marker: '{marker}'")
            print(f"  Linestyle: '{linestyle}'")
            print(f"  Color: {info['color']}")
            print(f"  Linewidth: {info['linewidth']}")
            print(f"  Current zorder: {info['zorder']}")
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nTotal collections: {diagnosis['summary']['total_collections']}")
        if diagnosis['summary']['collection_types']:
            print("\nCollection types found:")
            for ctype, count in diagnosis['summary']['collection_types'].items():
                print(f"  {ctype}: {count}")
        
        print(f"\nTotal lines: {diagnosis['summary']['total_lines']}")
        if diagnosis['summary']['marker_types']:
            print("\nMarker types found:")
            for marker, count in diagnosis['summary']['marker_types'].items():
                print(f"  '{marker}': {count}")
        
        if diagnosis['summary']['linestyle_types']:
            print("\nLinestyle types found:")
            for linestyle, count in diagnosis['summary']['linestyle_types'].items():
                print(f"  '{linestyle}': {count}")
        
        # Provide suggestions
        print("\n" + "=" * 70)
        print("RASTERIZATION SUGGESTIONS")
        print("=" * 70)
        
        if diagnosis['summary']['collection_types']:
            print("\nTo rasterize collections, use:")
            print("  rasterize_collections=[")
            for ctype in diagnosis['summary']['collection_types'].keys():
                print(f"      mcollections.{ctype},")
            print("  ]")
        
        if diagnosis['summary']['marker_types']:
            print("\nTo rasterize by marker type, use:")
            markers = list(diagnosis['summary']['marker_types'].keys())
            print(f"  rasterize_markers={markers}")
        
        if diagnosis['summary']['linestyle_types']:
            print("\nTo rasterize by linestyle, use:")
            linestyles = list(diagnosis['summary']['linestyle_types'].keys())
            print(f"  rasterize_linestyles={linestyles}")
        
        print("\n" + "=" * 70)
    
    return diagnosis

if __name__ == '__main__':

    # Hints
    #     -----
    #     Marker (rasterize_markers): The symbol used at data points (., o, s, ^, *, etc.)
    #     Linestyle (rasterize_linestyles): The style of the connecting line (-, --, -., :, etc.)
        
    #     PathCollection: scatter()
    #     PolyCollection: fill_between(), pcolormesh(), some bar charts
    #     LineCollection: plot() with multiple segments, eventplot()

    import matplotlib.pyplot as plt
    import matplotlib.collections as mcollections

    # Create various plot types
    x1 = np.random.uniform(0,10,10)
    y1 = np.random.uniform(0,10,10)

    x2 = np.random.uniform(0,10,10)
    y2 = np.random.uniform(10,20,10)

    x3 = np.random.uniform(0,10,10)
    y3 = np.random.uniform(20,30,10)

    x4 = np.random.uniform(0,10,10)
    y4a = 30
    y4b = 40

    # Example 0: Rasterize nothing
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, label='scatter')
    ax.plot(x2, y2, '.', label='points')
    ax.plot(x3, y3, '--', label='dashed')
    ax.fill_between(x4, y4a, y4b, alpha=0.3, label='fill')
    fig.savefig('example_0.pdf', dpi=30)

    # Example 1: Rasterize only scatter plots (PathCollection)
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, label='scatter')
    ax.plot(x2, y2, '.', label='points')
    ax.plot(x3, y3, '--', label='dashed')
    ax.fill_between(x4, y4a, y4b, alpha=0.3, label='fill')

    set_selective_rasterization(ax, 
                                rasterize_collections=[mcollections.PathCollection])
    plt.savefig('example_1.pdf', dpi=30)

    # Example 2: Rasterize plots with '.' marker (Line2D objects)
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, label='scatter')  # Won't be rasterized
    ax.plot(x2, y2, '.', label='points')  # Will be rasterized
    ax.plot(x3, y3, '--', label='dashed')  # Won't be rasterized
    ax.fill_between(x4, y4a, y4b, alpha=0.3, label='fill')  # Won't be rasterized

    set_selective_rasterization(ax, 
                                rasterize_markers=['.'],
                                rasterize_collections=[])  # Empty list = don't rasterize any collections
    plt.savefig('example_2.pdf', dpi=30)

    # Example 3: Rasterize BOTH scatter AND point markers
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, label='scatter')  # Will be rasterized
    ax.plot(x2, y2, '.', label='points')  # Will be rasterized
    ax.plot(x3, y3, '--', label='dashed')  # Won't be rasterized
    ax.fill_between(x4, y4a, y4b, alpha=0.3, label='fill')  # Won't be rasterized

    set_selective_rasterization(ax,
                                rasterize_markers=['.'],
                                rasterize_collections=[mcollections.PathCollection])
    plt.savefig('example_3.pdf', dpi=30)

    # Example 4: Rasterize both points and dashed markers, but not the scatter
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, label='scatter')  # Won't be rasterized
    ax.plot(x2, y2, '.', label='points')  # Will be rasterized
    ax.plot(x3, y3, '--', label='dashed')  # Will be rasterized
    ax.fill_between(x4, y4a, y4b, alpha=0.3, label='fill')  # Won't be rasterized

    set_selective_rasterization(ax,
                                rasterize_markers=['.'],
                                rasterize_linestyles=['--'],
                                rasterize_collections=[])
    plt.savefig('example_4.pdf', dpi=30)

    # Example 5: Rasterize the fill between
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, label='scatter')  # Won't be rasterized
    ax.plot(x2, y2, '.', label='points')  # Won't be rasterized
    ax.plot(x3, y3, '--', label='dashed')  # Won't be rasterized
    ax.fill_between(x4, y4a, y4b, alpha=0.3, label='fill', color='red')  # Will be rasterized

    set_selective_rasterization(ax,
                                rasterize_markers=[],
                                rasterize_collections=[mcollections.PolyCollection])
    plt.savefig('example_5.pdf', dpi=30)