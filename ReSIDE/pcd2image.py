import os

import plac

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


@plac.annotations(
    point_cloud_dir=plac.Annotation('The path to the point clouds.', abbrev='i', kind='option', type=str),
    orientation=plac.Annotation('Whether to put samples from the same test image along the same column or row.',
                                abbrev='o', kind='option', type=str, choices=['row', 'column'])
)
def main(point_cloud_dir, orientation='column'):
    decoders = sorted(filter(lambda x: os.path.isdir(os.path.join(point_cloud_dir, x)), os.listdir(point_cloud_dir)))
    encoders = sorted(os.listdir(os.path.join(point_cloud_dir, decoders[0])))

    collage = None

    vis = o3d.visualization.Visualizer()
    vis.create_window('pcl', 256, 256, 0, 0, True)
    view = vis.get_view_control()

    for decoder in decoders:
        for encoder in encoders:
            row = None

            for i in range(6):
                path = os.path.join(point_cloud_dir, decoder, encoder, f"{i:04d}.ply")

                pcd = o3d.io.read_point_cloud(path)

                vis.reset_view_point(True)

                vis.add_geometry(pcd)

                if i in {0, 1, 3}:
                    view.translate(x=-20, y=10)
                    view.rotate(x=-50, y=150)
                    view.set_zoom(0.3)
                else:
                    view.translate(x=10, y=0)
                    view.rotate(x=0, y=150)
                    view.rotate(x=-150, y=0)
                    view.set_zoom(0.2)

                if i == 3:
                    view.translate(x=0, y=-50)
                elif i == 5:
                    view.translate(x=-30, y=0)

                vis.update_renderer()

                out_image = vis.capture_screen_float_buffer(True)

                vis.remove_geometry(pcd)

                if row is None:
                    row = out_image
                else:
                    row = np.concatenate((row, out_image), axis=1 if orientation == 'column' else 0)

            if collage is None:
                collage = row
            else:
                collage = np.concatenate((collage, row), axis=0 if orientation == 'column' else 1)

    vis.close()
    vis.destroy_window()

    plt.imshow(collage)
    plt.show()

    image = Image.fromarray((255 * collage).astype(np.uint8))
    image.save("point_cloud_comparison.png")


if __name__ == '__main__':
    plac.call(main)
