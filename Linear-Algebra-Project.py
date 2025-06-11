
import matplotlib.pyplot as plt
import numpy as np

def apply_transformation(points, matrix):
    return np.dot(points, matrix.T)

def draw_hello_kitty(ax, transform_matrix=np.identity(2), title="Hello Kitty", show_axes=True):
    theta = np.linspace(0, 2 * np.pi, 100)

    # Head (oval)
    head = np.vstack((3 * np.cos(theta), 2.5 * np.sin(theta))).T

    # Eyes
    eye_left = np.vstack((0.4 * np.cos(theta) - 1, 0.4 * np.sin(theta))).T
    eye_right = np.vstack((0.4 * np.cos(theta) + 1, 0.4 * np.sin(theta))).T

    # Nose
    nose = np.vstack((0.2 * np.cos(theta), 0.2 * np.sin(theta) - 0.3)).T

    # Bow
    bow_left = np.vstack((0.5 * np.cos(theta) - 1.8, 0.5 * np.sin(theta) + 1.5)).T
    bow_right = np.vstack((0.5 * np.cos(theta) - 0.8, 0.5 * np.sin(theta) + 1.5)).T
    bow_center = np.vstack((0.2 * np.cos(theta) - 1.3, 0.2 * np.sin(theta) + 1.5)).T

    # Ears
    ear_left = np.array([[-2, 2], [-1.2, 4], [-0.5, 2]])
    ear_right = np.array([[0.5, 2], [1.2, 4], [2, 2]])

    # Whiskers
    whiskers = [
        ([-3, -1.5], [-0.5, -0.5]),
        ([-3, -1.5], [0, 0]),
        ([-3, -1.5], [0.5, 0.5]),
        ([1.5, 3], [-0.5, -0.5]),
        ([1.5, 3], [0, 0]),
        ([1.5, 3], [0.5, 0.5])
    ]

    # Apply transformation
    head = apply_transformation(head, transform_matrix)
    eye_left = apply_transformation(eye_left, transform_matrix)
    eye_right = apply_transformation(eye_right, transform_matrix)
    nose = apply_transformation(nose, transform_matrix)
    bow_left = apply_transformation(bow_left, transform_matrix)
    bow_right = apply_transformation(bow_right, transform_matrix)
    bow_center = apply_transformation(bow_center, transform_matrix)
    ear_left = apply_transformation(ear_left, transform_matrix)
    ear_right = apply_transformation(ear_right, transform_matrix)
    whiskers = [(apply_transformation(np.column_stack((x, y)), transform_matrix)) for x, y in whiskers]

    # Drawing
    ax.fill(head[:, 0], head[:, 1], color='white', edgecolor='black', linewidth=2)
    ax.fill(eye_left[:, 0], eye_left[:, 1], color='black')
    ax.fill(eye_right[:, 0], eye_right[:, 1], color='black')
    ax.fill(nose[:, 0], nose[:, 1], color='yellow')
    ax.fill(bow_left[:, 0], bow_left[:, 1], color='red')
    ax.fill(bow_right[:, 0], bow_right[:, 1], color='red')
    ax.fill(bow_center[:, 0], bow_center[:, 1], color='red')
    ax.fill(ear_left[:, 0], ear_left[:, 1], color='white', edgecolor='black', linewidth=2)
    ax.fill(ear_right[:, 0], ear_right[:, 1], color='white', edgecolor='black', linewidth=2)

    for whisker in whiskers:
        ax.plot(whisker[:, 0], whisker[:, 1], color='black', linewidth=1.5)

    ax.set_aspect('equal')
    ax.set_title(title)
    if show_axes:
        ax.grid(True)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    else:
        ax.axis('off')

# Transformation matrices for Group 6
compression_matrix = np.array([[1, 0], [0, 1/3]])
shear_matrix = np.array([[1, 4], [0, 1]])

# Plot all three side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

draw_hello_kitty(axes[0], np.identity(2), "Original Hello Kitty")
draw_hello_kitty(axes[1], compression_matrix, "Compressed (Y-axis × 1/3)")
draw_hello_kitty(axes[2], shear_matrix, "Sheared (X-axis × 4)")

plt.tight_layout()
plt.show()
