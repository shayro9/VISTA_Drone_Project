from PIL import Image, ImageDraw

# Create fully transparent image
width, height = 512, 512
img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

# Draw a red circle in the center
draw = ImageDraw.Draw(img)
radius = 100
center = (width // 2, height // 2)
bounding_box = [  # Circle bounds
    (center[0] - radius, center[1] - radius),
    (center[0] + radius, center[1] + radius),
]
draw.ellipse(bounding_box, fill=(255, 0, 0, 255))  # Red with full opacity

# Save
img.save("circle_in_transparent.png")
