# coding: utf-8
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def plot_geo_rgb(rgb:np.ndarray, lat_range:tuple, lon_range:tuple,
        plot_spec:dict={}, fig_path=None, show=False):
    """ """
    ps = {"title":"", "figsize":(16,12), "border_linewidth":2,
            "title_size":12 }
    ps.update(plot_spec)
    fig = plt.figure(figsize=ps.get("figsize"))

    pc = ccrs.PlateCarree()

    ax = fig.add_subplot(1, 1, 1, projection=pc)
    extent = [*lon_range, *lat_range]
    ax.set_extent(extent, crs=pc)

    ax.imshow(rgb, extent=extent, transform=pc)
    print(extent)

    ax.coastlines(
            color=ps.get("border_color", "black"),
            linewidth=ps.get("border_linewidth"))
    ax.add_feature(
            ccrs.cartopy.feature.STATES,
            #color=ps.get("border_color", "black"),
            linewidth=ps.get("border_linewidth")
            )

    plt.title(ps.get("title"), fontweight='bold',
            fontsize=ps.get("title_size"))

    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)
    if show:
        plt.show()
    plt.close()
    return

if __name__=="__main__":
    slabels,sdata = pkl.load(Path("data/static/era5_static.pkl").open("rb"))
    lm = sdata[slabels.index("landmask")]
    m_valid = sdata[slabels.index("m_valid")].astype(bool)

    fig_dir = Path("/rhome/mdodson/emulate-era5-land/figures/landmask")
    for i in range(21):
        ratio = i*.05
        tmpm = (lm < ratio) & m_valid
        rgb = np.where(tmpm, 255, 0)
        rgb = np.stack([rgb for i in range(3)], axis=-1)
        plot_geo_rgb(
            rgb=rgb,
            lat_range=(24,50),
            lon_range=(-125,-66.5),
            plot_spec={
                "border_color":"black",
                "border_linewidth":1,
                "title":f"landmask < {ratio}"
                },
            fig_path=fig_dir.joinpath(f"landmask_{int(ratio*100):03}.png"),
            show=False
            )
