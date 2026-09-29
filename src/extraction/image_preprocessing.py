import os
import cv2


def preprocess_image(image_path: str) -> str:

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    print("Original image shape:", image.shape)

    # Relative crop
    top = int(height * 0.20)
    bottom = int(height * 0.60)

    left = int(width * 0.10)
    right = int(width * 0.90)

    cropped = image[
        top:bottom,
        left:right
    ]

    # Upscale
    resized = cv2.resize(
        cropped,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    output_path = os.path.join(
        "src",
        "test_images",
        "resized_image.jpg"
    )

    cv2.imwrite(
        output_path,
        resized
    )

    print(
        "Processed image shape:",
        resized.shape
    )

    print(
        "Preprocessed image saved to:",
        output_path
    )

    return output_path