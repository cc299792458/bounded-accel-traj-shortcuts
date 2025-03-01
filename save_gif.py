import os
import imageio

def save_gif_from_frames(frames_folder="smoothing_frames", output_gif="trajectory_smoothing.gif"):
    """
    Create a GIF from saved frames.
    
    Args:
        frames_folder (str): Path to the folder where frames are stored.
        output_gif (str): Output path for the gif file.
    """
    images = []
    for frame_file in sorted(os.listdir(frames_folder)):
        if frame_file.endswith(".png"):
            frame_path = os.path.join(frames_folder, frame_file)
            images.append(imageio.imread(frame_path))

    imageio.mimsave(output_gif, images, duration=300, loop=0)  # Adjust the duration as needed


if __name__ == '__main__':
    save_gif_from_frames()