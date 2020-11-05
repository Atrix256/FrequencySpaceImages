#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <direct.h>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "simple_fft/fft_settings.h"
#include "simple_fft/fft.h"

#include <vector>

static const float c_pi = 3.14159265359f;

inline bool IsPowerOf2(size_t n)
{
    return (n & (n - 1)) == 0;
}

inline size_t NextPowerOf2(size_t n)
{
    size_t ret = 1;
    while (ret < n)
        ret = ret << 1;
    return ret;
}

template <typename T>
T Clamp(T value, T min, T max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

float Lerp(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

struct ComplexImage2D
{
    ComplexImage2D(size_t w = 0, size_t h = 0)
    {
        Resize(w, h);
    }

    size_t m_width;
    size_t m_height;
    std::vector<complex_type> pixels;

    void Resize(size_t w, size_t h)
    {
        m_width = w;
        m_height = h;
        pixels.resize(w * h, real_type(0.0f));
    }

    complex_type& operator()(size_t x, size_t y)
    {
        return pixels[y * m_width + x];
    }

    const complex_type& operator()(size_t x, size_t y) const
    {
        return pixels[y * m_width + x];
    }

    void IndexToCoordinates(size_t index, size_t& x, size_t& y) const
    {
        y = index / m_width;
        x = index % m_width;
    }
};

void SaveDFTMagPhaseIDFT(const ComplexImage2D& image, const char* baseFileName)
{
    char buffer[4096];

    // write the .dft file so we can see how it compresses
    {
        strcpy(buffer, baseFileName);
        strcat(buffer, ".dft");

        FILE* file = nullptr;
        fopen_s(&file, buffer, "wb");
        fwrite(image.pixels.data(), image.m_width * image.m_height * sizeof(double) * 2, 1, file);
        fclose(file);
    }

    // write the magnitude and phase images - with dc in the middle, instead of at (0,0)
    {
        std::vector<float> mag(image.pixels.size());
        std::vector<float> phase(image.pixels.size());

        float maxMag = 0.0f;
        for (size_t index = 0; index < image.pixels.size(); ++index)
        {
            size_t px, py;
            image.IndexToCoordinates(index, px, py);
            px = (px + image.m_width / 2) % image.m_width;
            py = (py + image.m_height / 2) % image.m_height;
            const complex_type& c = image(px, py);

            mag[index] = float(log(1.0f + float(sqrt(c.real() * c.real() + c.imag() * c.imag()))));
            phase[index] = float(atan2(c.imag(), c.real()));
            maxMag = std::max(maxMag, mag[index]);
        }

        // write magnitude image
        std::vector<uint8_t> pixels(image.pixels.size());
        for (size_t index = 0; index < image.pixels.size(); ++index)
            pixels[index] = (uint8_t)Clamp(mag[index] / maxMag * 256.0f, 0.0f, 255.0f);
        strcpy(buffer, baseFileName);
        strcat(buffer, ".mag.png");
        stbi_write_png(buffer, (int)image.m_width, (int)image.m_height, 1, pixels.data(), (int)image.m_width);

        // write phase image
        for (size_t index = 0; index < image.pixels.size(); ++index)
        {
            float value = (phase[index] + c_pi) / (2.0f * c_pi);
            pixels[index] = (uint8_t)Clamp(value * 256.0f, 0.0f, 255.0f);
        }
        strcpy(buffer, baseFileName);
        strcat(buffer, ".phase.png");
        stbi_write_png(buffer, (int)image.m_width, (int)image.m_height, 1, pixels.data(), (int)image.m_width);
    }

    // IDFT the image and write it out
    {
        // IDFT the image
        ComplexImage2D complexImageOut;
        {
            const char* error = nullptr;
            complexImageOut.Resize(image.m_width, image.m_height);
            simple_fft::IFFT(image, complexImageOut, image.m_width, image.m_height, error);
        }

        // convert to U8 and write it out
        {
            std::vector<uint8_t> pixels;
            pixels.resize(complexImageOut.pixels.size());

            for (size_t index = 0; index < complexImageOut.pixels.size(); ++index)
            {
                const complex_type& c = complexImageOut.pixels[index];
                float mag = float(sqrt(c.real() * c.real() + c.imag() * c.imag()));

                pixels[index] = (uint8_t)Clamp(mag * 256.0f, 0.0f, 255.0f);
            }

            strcpy(buffer, baseFileName);
            strcat(buffer, ".png");
            stbi_write_png(buffer, (int)complexImageOut.m_width, (int)complexImageOut.m_height, 1, pixels.data(), (int)complexImageOut.m_width);
        }
    }
}

void DoTestZeroing(const ComplexImage2D& _imageDFT, const char* fileName)
{
    char buffer[4096];

    // make a copy since this is a destructive process
    ComplexImage2D imageDFT = _imageDFT;

    // get a list of pixels sorted by magnitude
    struct PixelMagnitude
    {
        size_t index;
        float magnitude;
    };
    std::vector<PixelMagnitude> pixelMagnitudes(imageDFT.pixels.size());
    {
        for (size_t index = 0; index < imageDFT.pixels.size(); ++index)
        {
            pixelMagnitudes[index].index = index;
            const complex_type& c = imageDFT.pixels[index];
            pixelMagnitudes[index].magnitude = float(sqrt(c.real() * c.real() + c.imag() * c.imag()));
        }
        std::sort(
            pixelMagnitudes.begin(),
            pixelMagnitudes.end(),
            [] (const PixelMagnitude& A, const PixelMagnitude& B)
            {
                return A.magnitude < B.magnitude;
            }
        );
    }

    // throw away lowest X% magnitude of DFT data and save that out, along with an idft made image
    {
        // first save a data for the unmodified image, for comparisons
        {
            sprintf(buffer, "out/%s.cull.0", fileName);
            SaveDFTMagPhaseIDFT(imageDFT, buffer);
        }
        // now throw away the lowest magnitude frequencies
        for (size_t percentIndex = 1; percentIndex < 10; ++percentIndex)
        {
            // calculate the index range we want to zero out
            float percent = float(percentIndex) / float(10.0f);
            size_t lastIndex = size_t(float(pixelMagnitudes.size()) * percent);

            // set them all to zero
            for (size_t index = 0; index < lastIndex; ++index)
                imageDFT.pixels[pixelMagnitudes[index].index] = complex_type();

            sprintf(buffer, "out/%s.cull.%i", fileName, (int)percentIndex);
            SaveDFTMagPhaseIDFT(imageDFT, buffer);
        }
    }
}

void DoTestFiltering(const ComplexImage2D& imageDFT, const char* fileName)
{
    char buffer[4096];

    ComplexImage2D imageDFT_HPF5 = imageDFT;
    ComplexImage2D imageDFT_HPF10 = imageDFT;
    ComplexImage2D imageDFT_LPF5 = imageDFT;
    ComplexImage2D imageDFT_LPF10 = imageDFT;
    ComplexImage2D imageDFT_Notch = imageDFT;
    ComplexImage2D imageDFT_BandPass = imageDFT;

    int width = (int)imageDFT.m_width;
    int height = (int)imageDFT.m_height;

    int halfWidth = width / 2;
    int halfHeight = height / 2;

    std::vector<uint8_t> filter_hpf5(imageDFT.m_width * imageDFT.m_height);
    std::vector<uint8_t> filter_hpf10(imageDFT.m_width * imageDFT.m_height);
    std::vector<uint8_t> filter_lpf5(imageDFT.m_width * imageDFT.m_height);
    std::vector<uint8_t> filter_lpf10(imageDFT.m_width * imageDFT.m_height);
    std::vector<uint8_t> filter_notch(imageDFT.m_width * imageDFT.m_height);
    std::vector<uint8_t> filter_bandpass(imageDFT.m_width * imageDFT.m_height);

    for (size_t index = 0; index < imageDFT.pixels.size(); ++index)
    {
        size_t px, py;
        imageDFT.IndexToCoordinates(index, px, py);

        int ipx = int((px + halfWidth) % width) - halfWidth;
        int ipy = int((py + halfHeight) % height) - halfHeight;

        float percentx = float(2 * ipx) / float(imageDFT.m_width);
        float percenty = float(2 * ipy) / float(imageDFT.m_height);

        float distance = sqrt(percentx * percentx + percenty * percenty) / sqrt(2.0f);
        distance = Clamp(distance, 0.0f, 1.0f);
        float invDist = 1.0f - distance;

        float lpf5 = pow(invDist, 5.0f);
        float lpf10 = pow(invDist, 10.0f);
        float hpf5 = 1.0f - pow(invDist, 5.0f);
        float hpf10 = 1.0f - pow(invDist, 10.0f);

        float notch = abs(distance - 0.5f) * 2.0f;
        float bandpass = 1.0f - notch;

        imageDFT_HPF5.pixels[index] *= hpf5;
        imageDFT_HPF10.pixels[index] *= hpf10;
        imageDFT_LPF5.pixels[index] *= lpf5;
        imageDFT_LPF10.pixels[index] *= lpf10;
        imageDFT_Notch.pixels[index] *= notch;
        imageDFT_BandPass.pixels[index] *= bandpass;

        ipx += halfWidth;
        ipy += halfHeight;
        filter_hpf5[ipy * width + ipx] = (uint8_t)Clamp(hpf5 * 256.0f, 0.0f, 255.0f);
        filter_hpf10[ipy * width + ipx] = (uint8_t)Clamp(hpf10 * 256.0f, 0.0f, 255.0f);
        filter_lpf5[ipy * width + ipx] = (uint8_t)Clamp(lpf5 * 256.0f, 0.0f, 255.0f);
        filter_lpf10[ipy * width + ipx] = (uint8_t)Clamp(lpf10 * 256.0f, 0.0f, 255.0f);
        filter_notch[ipy * width + ipx] = (uint8_t)Clamp(notch * 256.0f, 0.0f, 255.0f);
        filter_bandpass[ipy * width + ipx] = (uint8_t)Clamp(bandpass * 256.0f, 0.0f, 255.0f);
    }

    sprintf(buffer, "out/%s.hpf5", fileName);
    SaveDFTMagPhaseIDFT(imageDFT_HPF5, buffer);

    sprintf(buffer, "out/%s.hpf10", fileName);
    SaveDFTMagPhaseIDFT(imageDFT_HPF10, buffer);

    sprintf(buffer, "out/%s.lpf5", fileName);
    SaveDFTMagPhaseIDFT(imageDFT_LPF5, buffer);

    sprintf(buffer, "out/%s.lpf10", fileName);
    SaveDFTMagPhaseIDFT(imageDFT_LPF10, buffer);

    sprintf(buffer, "out/%s.notch", fileName);
    SaveDFTMagPhaseIDFT(imageDFT_Notch, buffer);

    sprintf(buffer, "out/%s.bandpass", fileName);
    SaveDFTMagPhaseIDFT(imageDFT_BandPass, buffer);

    // save the filters for visualization purposes
    stbi_write_png("out/filter_hpf5.png", width, height, 1, filter_hpf5.data(), width);
    stbi_write_png("out/filter_hpf10.png", width, height, 1, filter_hpf10.data(), width);
    stbi_write_png("out/filter_lpf5.png", width, height, 1, filter_lpf5.data(), width);
    stbi_write_png("out/filter_lpf10.png", width, height, 1, filter_lpf10.data(), width);
    stbi_write_png("out/filter_notch.png", width, height, 1, filter_notch.data(), width);
    stbi_write_png("out/filter_bandpass.png", width, height, 1, filter_bandpass.data(), width);
}

bool LoadImage(const char* fileName, ComplexImage2D& image)
{
    // load the image
    int w, h, comp;
    stbi_uc* pixels = stbi_load(fileName, &w, &h, &comp, 1);
    if (!pixels)
    {
        printf(__FUNCTION__ "(): Could not load image %s", fileName);
        return false;
    }

    // convert to complex pixels
    image.Resize(w, h);
    for (size_t index = 0; index < image.pixels.size(); ++index)
        image.pixels[index] = float(pixels[index]) / 255.0f;

    // free the pixels
    stbi_image_free(pixels);

    // verify the image is a power of 2
    if (!IsPowerOf2(w) || !IsPowerOf2(h))
    {
        printf(__FUNCTION__ "(): image is %ix%i but width and height need to be a power of 2", w, h);
        return false;
    }

    return true;
}

ComplexImage2D ZeroPad(const ComplexImage2D& image, size_t width, size_t height)
{
    size_t offsetx = (width - image.m_width) / 2;
    size_t offsety = (height - image.m_height) / 2;

    ComplexImage2D ret(width, height);

    for (size_t iy = 0; iy < image.m_height; ++iy)
    {
        const complex_type* src = &image.pixels[iy * image.m_width];
        complex_type* dest = &ret.pixels[(iy + offsety) * ret.m_width + offsetx];
        memcpy(dest, src, sizeof(complex_type) * image.m_width);
    }

    return ret;
}

ComplexImage2D UnZeroPad(const ComplexImage2D& image, size_t width, size_t height)
{
    size_t offsetx = (image.m_width - width) / 2;
    size_t offsety = (image.m_height - height) / 2;

    ComplexImage2D ret(width, height);

    for (size_t iy = 0; iy < height; ++iy)
    {
        const complex_type* src = &image.pixels[(iy + offsety) * image.m_width + offsetx];
        complex_type* dest = &ret.pixels[iy * ret.m_width];
        memcpy(dest, src, sizeof(complex_type) * width);
    }

    return ret;
}

ComplexImage2D ShiftImage(const ComplexImage2D& image, size_t offsetx, size_t offsety)
{
    ComplexImage2D ret = image;

    for (size_t iy = 0; iy < image.m_height; ++iy)
    {
        size_t srciy = (iy + offsety) % image.m_height;
        for (size_t ix = 0; ix < image.m_width; ++ix)
        {
            size_t srcix = (ix + offsetx) % image.m_width;
            ret(ix, iy) = image(srcix, srciy);
        }
    }

    return ret;
}

void Normalize(ComplexImage2D& image)
{
    double sum = 0.0f;
    for (const complex_type& c : image.pixels)
        sum += sqrt(c.imag() * c.imag() + c.real() * c.real());

    for (complex_type& c : image.pixels)
        c /= sum;
}

void DoTestConvolution(ComplexImage2D image, const char* fileName, const char* kernelName)
{
    // load the star - what we are going to use for convolution
    char buffer[4096];
    sprintf(buffer, "assets/%s.png", kernelName);
    ComplexImage2D imageKernel;
    LoadImage(buffer, imageKernel);

    // make sure the kernel sums to 1.0
    Normalize(imageKernel);

    size_t initialWidth = image.m_width;
    size_t initialHeight = image.m_height;

    // calculate the size that the images need to be, to be multiplied in frequency space
    size_t desiredWidth = NextPowerOf2(image.m_width + imageKernel.m_width + 1);
    size_t desiredHeight = NextPowerOf2(image.m_height + imageKernel.m_height + 1);

    // zero pad the images to be the right size
    image = ZeroPad(image, desiredWidth, desiredHeight);
    imageKernel = ZeroPad(imageKernel, desiredWidth, desiredHeight);

    // Need to shift the star kernel half an image over: https://stackoverflow.com/questions/54877892/convolving-image-with-kernel-in-fourier-domain
    // (0,0) is the center when convolving, so need to fix that to be true
    imageKernel = ShiftImage(imageKernel, imageKernel.m_width / 2, imageKernel.m_height / 2);

    // DFT the images
    ComplexImage2D imageKernelDFT(desiredWidth, desiredHeight), imageDFT(desiredWidth, desiredHeight);
    const char* error = nullptr;
    simple_fft::FFT(image, imageDFT, image.m_width, image.m_height, error);
    simple_fft::FFT(imageKernel, imageKernelDFT, imageKernel.m_width, imageKernel.m_height, error);

    // multiply in frequency domain, to convolve the images
    ComplexImage2D resultDFT(desiredWidth, desiredHeight);
    for (size_t index = 0; index < imageDFT.pixels.size(); ++index)
        resultDFT.pixels[index] = imageDFT.pixels[index] * imageKernelDFT.pixels[index];

    // IDFT the result
    ComplexImage2D result(desiredWidth, desiredHeight);
    simple_fft::IFFT(resultDFT, result, resultDFT.m_width, resultDFT.m_height, error);

    // remove the extra bordering we put on
    result = UnZeroPad(result, initialWidth, initialHeight);

    // convert to U8
    std::vector<uint8_t> pixels(result.pixels.size());
    for (size_t index = 0; index < result.pixels.size(); ++index)
        pixels[index] = (uint8_t)Clamp(result.pixels[index].real() * 256.0, 0.0, 255.0);

    // write out the result
    sprintf(buffer, "out/%s.conv.%s.png", fileName, kernelName);
    stbi_write_png(buffer, (int)result.m_width, (int)result.m_height, 1, pixels.data(), (int)result.m_width);
}

void DoTests(const char* fileName)
{
    char buffer[4096];

    // load image and dft it
    ComplexImage2D image, imageDFT;
    {
        // load the image
        strcpy(buffer, "assets/");
        strcat(buffer, fileName);
        LoadImage(buffer, image);

        // DFT the image
        const char* error = nullptr;
        imageDFT.Resize(image.m_width, image.m_height);
        simple_fft::FFT(image, imageDFT, image.m_width, image.m_height, error);
    }

    // test zeroing out low magnitude frequencies
    DoTestZeroing(imageDFT, fileName);

    // do high pass and low pass filtering by attenuating magnitudes based on distance from center (DC / 0hz)
    DoTestFiltering(imageDFT, fileName);

    // convolve the images against other images
    const char* files[] =
    {
        "star",
        "plus",
        "circle",
        "blob"
    };

    for (size_t index = 0; index < _countof(files); ++index)
        DoTestConvolution(image, fileName, files[index]);
}

int main(int argc, char** argv)
{
    _mkdir("out");

    const char* files[] =
    {
        "lokialan.jpg",
        "scenery.png",
        "star.png",
        "plus.png",
        "circle.png",
        "blob.png",
        "BlueNoise.png",
    };

    for (size_t index = 0; index < _countof(files); ++index)
        DoTests(files[index]);

    return 0;
}
