#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint>

// ---- Parámetros ---------------------------------------------------------- //

const int    W        = 3840;
const int    H        = 2160;
const int    MAX_ITER = 1000;

// Vista completa
const double XMIN = -2.5;
const double XMAX =  1.0;
const double YMIN = -1.1;
const double YMAX =  1.1;

// Seahorse Valley (aumentar MAX_ITER a 2000)
// const double XMIN = -0.751; const double XMAX = -0.741;
// const double YMIN =  0.100; const double YMAX =  0.110;

// Elephant Valley (aumentar MAX_ITER a 2000)
// const double XMIN = 0.27052033; const double XMAX = 0.27105033;
// const double YMIN = 0.00560804; const double YMAX = 0.00591804;

// ---- Paleta de colores (no modificar) ------------------------------------ //

struct RGB { unsigned char r, g, b; };

RGB palette(double t) {
    struct Stop { double t; RGB c; };
    static const Stop stops[] = {
        {0.000, {  0,   7, 100}},
        {0.160, { 32, 107, 203}},
        {0.420, {237, 255, 255}},
        {0.642, {255, 170,   0}},
        {0.857, {  0,   2,   0}},
        {1.000, {  0,   7, 100}},
    };
    const int n = 6;
    for (int i = 0; i < n - 1; i++) {
        if (t <= stops[i + 1].t) {
            double s = (t - stops[i].t) / (stops[i + 1].t - stops[i].t);
            return {
                (unsigned char)(stops[i].c.r + s * (stops[i + 1].c.r - stops[i].c.r)),
                (unsigned char)(stops[i].c.g + s * (stops[i + 1].c.g - stops[i].c.g)),
                (unsigned char)(stops[i].c.b + s * (stops[i + 1].c.b - stops[i].c.b))
            };
        }
    }
    return stops[n - 1].c;
}

RGB color(int iter, double zr, double zi) {
    if (iter == MAX_ITER) return {0, 0, 0};
    double log_zn = 0.5 * std::log(zr * zr + zi * zi);
    double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
    double t      = std::fmod(((double)iter + 1.0 - nu) / 50.0, 1.0);
    return palette(t);
}

// ---- Escritura BMP ------------------------------------------------------- //

void save_bmp(const std::vector<RGB>& img, const std::string& path) {
    const int stride    = (W * 3 + 3) & ~3;
    const int px_size   = stride * H;
    const int file_size = 54 + px_size;

    auto le32 = [](std::ofstream& f, int32_t v) {
        uint32_t u = (uint32_t)v;
        f.put(u & 0xFF); f.put((u >> 8) & 0xFF);
        f.put((u >> 16) & 0xFF); f.put((u >> 24) & 0xFF);
    };
    auto le16 = [](std::ofstream& f, int16_t v) {
        f.put(v & 0xFF); f.put((v >> 8) & 0xFF);
    };

    std::ofstream f(path, std::ios::binary);
    f << "BM";
    le32(f, file_size); le32(f, 0); le32(f, 54);   // file header
    le32(f, 40); le32(f, W); le32(f, -H);           // DIB header
    le16(f, 1); le16(f, 24);                        // planes, bpp
    le32(f, 0); le32(f, px_size);                   // compression, size
    le32(f, 2835); le32(f, 2835);                   // ppm (~72 dpi)
    le32(f, 0); le32(f, 0);                         // color table

    std::vector<unsigned char> row(stride, 0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const RGB& p  = img[y * W + x];
            row[x * 3]    = p.b;
            row[x * 3 + 1] = p.g;
            row[x * 3 + 2] = p.r;
        }
        f.write(reinterpret_cast<char*>(row.data()), stride);
    }
}

// ---- Loop principal ------------------------------------------------------ //

int main() {
    
    std::vector<RGB> img(W * H);

    double zr, zi, cr, ci;
    int iter;
    
    #pragma omp parallel for schedule(dynamic) private(zr, zi, cr, ci, iter)
    for (int row = 0; row < H; row++) {
        ci = YMIN + (YMAX - YMIN) * row / (H - 1);
        for (int col = 0; col < W; col++) {
            cr   = XMIN + (XMAX - XMIN) * col / (W - 1);
            zr   = 0.0;
            zi   = 0.0;
            iter = 0;
            while (iter < MAX_ITER && zr * zr + zi * zi < 4.0) {
                double tmp = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = tmp;
                iter++;
            }
            img[row * W + col] = color(iter, zr, zi);
        }
    }

    save_bmp(img, "mandelbrot.bmp");
    std::cout << "Imagen guardada en mandelbrot.bmp\n";
    return 0;
}
