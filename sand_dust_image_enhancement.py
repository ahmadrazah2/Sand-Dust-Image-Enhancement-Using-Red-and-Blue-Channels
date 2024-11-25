import numpy as np
import cv2
import time


class SandDustEnhancement:
    def __init__(self, k=2, window_size=15, omega=0.95, guide_eps=0.001):
        """
        Initialize parameters for dual-channel enhancement
        """
        self.k = k
        self.window_size = window_size
        self.omega = omega
        self.guide_eps = guide_eps
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def rcc_module(self, image):
        """
        Red Channel Correction Function Module
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img_rgb)

        mu_r = np.mean(r)
        mu_g = np.mean(g)
        mu_b = np.mean(b)
        # Step 1: Channel translation according to equation (5)
        # Red channel remains unchanged
        # Adjust green and blue channels
        # based on difference with red channel mean
        g_translated = g + (mu_r - mu_g)
        b_translated = b + (mu_r - mu_b)

        # Step 2: Color stretching in RGB space according to equation (6)
        # Calculate variance for each channel
        sigma_r = np.std(r)
        sigma_g = np.std(g)
        sigma_b = np.std(b)

        # Apply stretching to each channel
        r_corrected = self.stretch_channel(r, mu_r, sigma_r)
        g_corrected = self.stretch_channel(g_translated, mu_r, sigma_g)
        b_corrected = self.stretch_channel(b_translated, mu_r, sigma_b)

        # Merge the channels back
        corrected_image = cv2.merge([r_corrected, g_corrected, b_corrected])

        return cv2.cvtColor(corrected_image , cv2.COLOR_RGB2BGR)

    def stretch_channel(self, channel, mu_c, sigma_c):
        # Calculate I_max and I_min
        I_max = mu_c + self.k * sigma_c
        I_min = mu_c - self.k * sigma_c

        # Clip I_max and I_min to valid range [0, 255]
        I_max = np.clip(I_max, 0, 255)
        I_min = np.clip(I_min, 0, 255)

        # Apply stretching formula
        stretched = 255 * (channel - I_min) / (I_max - I_min + 1e-8)
        # 1e-8 adding small epsilon to avoid division by zero
        return np.clip(stretched, 0, 255).astype(np.uint8)

    def apply_clahe(self, image):
        """
        Contrast-limited Adaptive Histogram Equalization
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)

        # Merge back
        enhanced = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def estimate_atmospheric_light(self,image):
        """
        Atmospheric Light Estimation following the paper more closely
        """
        img = image.astype(np.float32) / 255.0
        # Split channels (BGR format)
        b, g, r = cv2.split(img)
        # Calculate normalized blue reflectance
        epsilon = 1e-6  # Avoid division by zero
        max_intensity = np.maximum(np.maximum(b, g), r)
        normalized_blue = b / (max_intensity + epsilon)
        # Calculate modified dark channel components according to equation (9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        # min(R^B(y)): Minimum of blue channel
        min_blue = cv2.erode(b, kernel)
        # min(R^G(y)): Minimum of green channel
        min_green = cv2.erode(g, kernel)
        # min(1-R^N(y)): Minimum of inverse normalized blue
        min_inv_norm = cv2.erode(1 - normalized_blue, kernel)
        # Combine according to equation (9)
        dark_channel = np.minimum(np.minimum(min_blue, min_green), min_inv_norm)
        # Get size and number of pixels to consider (0.1%)
        size = dark_channel.shape[0] * dark_channel.shape[1]
        num_pixels = int(max(size * 0.001, 1))
        # Find brightest pixels in  dark channel
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        # Among these pixels, find the one with highest blue channel intensity
        max_blue = -1
        atmospheric_light = np.zeros(3, dtype=np.float32)
        for idx in indices:
            y = idx // dark_channel.shape[1]
            x = idx % dark_channel.shape[1]
            pixel = img[y, x]

            # Consider both blue intensity and normalized blue reflectance
            blue_score = pixel[0] * (1 - normalized_blue[y, x])

            if blue_score > max_blue:
                max_blue = blue_score
                atmospheric_light = pixel

        return (atmospheric_light * 255).astype(np.uint8)


    def calculate_dark_bright_channels(self, img_rcc):
        """
        Calculate dark and bright channels (Equation 10)
        """
        img_rgb = cv2.cvtColor(img_rcc, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img_rgb)
        inverse_b = 1 - b

        dark_channel = np.minimum.reduce([r, g, inverse_b])
        bright_channel = np.maximum.reduce([r, g, inverse_b])

        return dark_channel, bright_channel

    def estimate_transmission(self, img_rcc, dark_channel, bright_channel):
        """
        Estimate transmission map (Equations 11 and 12)
        """
        # Calculate sand dust density (Equation 11)
        bright_dark_diff = bright_channel - dark_channel
        max_bright = np.maximum(1, bright_channel)
        density = dark_channel * (1 - bright_dark_diff / max_bright)
        # Apply minimum filter
        density = cv2.erode(density,np.ones((self.window_size,
                                             self.window_size), np.float32))
        # Refine using guided filter
        gray_guide = cv2.cvtColor(img_rcc,
                                  cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
        density_refined = self.guided_filter(gray_guide, density,
                                             self.window_size, self.guide_eps)

        # Calculate transmission (Equation 12)
        transmission = 1 - self.omega * density_refined
        return  np.clip(transmission, 0.1, 1)

    def guided_filter(self, guide, src, radius, eps):
        """
        Guided Filter Implementation
        """
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)

        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_src = cv2.boxFilter(src, -1, (radius, radius))
        corr_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
        corr_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))

        var_guide = corr_guide - mean_guide * mean_guide
        cov_guide_src = corr_guide_src - mean_guide * mean_src

        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide + mean_b

    def detail_recovery(self, image, transmission, atmospheric_light):
        """
        Detail Recovery using Transmission Map
        """
        result = np.empty_like(image, dtype=np.float32)

        for i in range(3):
            result[:, :, i] = ((image[:, :, i].astype(np.float32) - atmospheric_light[i]) /
                               transmission + atmospheric_light[i])

        return np.clip(result, 0, 255).astype(np.uint8)

    def fusion(self, clahe_result, detail_recovery_result):
        """
        Final Fusion of CLAHE and Detail Recovery Results
        """
        # Simple weighted fusion
        weight_clahe = 0.4
        weight_detail = 0.2
        fused = cv2.addWeighted(clahe_result, weight_clahe,
                                detail_recovery_result, weight_detail,0)
        return fused

    def enhance_image(self, image):
        """
        Complete Enhancement Pipeline Following the Flowchart
        """
        # 1. Red Channel Correction
        rcc_result = self.rcc_module(image)

        # 2. CLAHE Enhancement
        clahe_result_rcc = self.apply_clahe(rcc_result)

        # 3. Blue Channel Sandstorm Removal
        # 3.1 Atmospheric Light Estimation
        atmospheric_light = self.estimate_atmospheric_light(image)

        dark_channel, bright_channel = self.calculate_dark_bright_channels(rcc_result)
        # cv2.imshow('dark' , dark_channel)
        # cv2.waitKey()
        # Step 3: Estimate atmospheric light


        # Step 4: Estimate transmission map
        transmission = self.estimate_transmission(rcc_result, dark_channel, bright_channel)




        # 3.4 Detail Recovery
        detail_recovery_result = self.detail_recovery(image, transmission, atmospheric_light)
        clahe_result_detail_recovery_result = self.apply_clahe(detail_recovery_result)
        # 4. Final Fusion
        final_result = self.fusion(clahe_result_rcc, clahe_result_detail_recovery_result)

        return final_result


def main():
    # Initialize enhancer
    enhancer = SandDustEnhancement()

    # Read image
    image_path = 'dark.jpeg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Process image
    start_time = time.time()
    result = enhancer.enhance_image(image)
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")

    cv2.imshow('Original', image)
    cv2.imshow('Enhanced', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()