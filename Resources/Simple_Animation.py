from PIL import Image, ImageEnhance
import imageio
import os

# Load the image
img = Image.open("PhishShield.jpg").convert("RGBA")
frames = []

# Create 10 frames for the animation
for i in range(10):
    scale = 1 + 0.02 * (5 - abs(5 - i))  # pulse in and out
    new_size = (int(img.width * scale), int(img.height * scale))
    frame = img.resize(new_size, Image.LANCZOS)

    # Create a white background and paste resized logo centered
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    offset = ((img.width - frame.width) // 2, (img.height - frame.height) // 2)
    bg.paste(frame, offset, frame)

    frames.append(bg)

# Save as GIF
frames[0].save("PhishShield_animated.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
