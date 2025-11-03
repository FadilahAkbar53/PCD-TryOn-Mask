# Try-On Mask - Godot Client

Client Godot untuk menerima dan menampilkan stream video dari Python server dengan deteksi wajah dan mask overlay.

## Requirements

- Godot Engine 4.3 atau lebih baru
- Python server (`godot_server.py`) yang sudah berjalan

## Cara Menggunakan

### 1. Jalankan Python Server

Di terminal/PowerShell, jalankan:

```powershell
python godot_server.py --camera 0
```

Server akan:

- Membuka webcam
- Mendeteksi wajah menggunakan SVM+ORB
- Menerapkan mask overlay
- Mengirim hasil ke Godot via UDP (port 5005)

### 2. Buka Godot Client

1. Buka Godot Engine
2. Import project ini (`godot_client` folder)
3. Tekan F5 atau klik "Run Project"

Client akan:

- Mendengarkan di port 5005
- Menerima frame dari Python server
- Menampilkan hasil di window Godot

## Konfigurasi

### Mengubah Server Address/Port

Edit file `tryon_client.gd`:

```gdscript
var server_address := "127.0.0.1"  # Ubah jika server di komputer lain
var server_port := 5005            # Ubah sesuai port server
```

### Python Server Options

```powershell
# Kualitas JPEG (1-100, default: 80)
python godot_server.py --quality 90

# Ukuran maksimal frame (default: 640)
python godot_server.py --max_size 800

# Port berbeda
python godot_server.py --port 6000

# Tanpa preview window
python godot_server.py --no_preview
```

## Controls

Kontrol dilakukan di sisi Python server:

- **Q** - Quit/keluar
- **M** - Toggle mask on/off
- **1-7** - Ganti mask (mask1.png - mask7.png)
- **S** - Screenshot

## Troubleshooting

### "Failed to bind port"

- Port 5005 sudah digunakan aplikasi lain
- Tutup aplikasi lain atau ubah port di kedua sisi (server & client)

### "No frames received"

- Pastikan Python server sudah running
- Cek firewall (izinkan UDP port 5005)
- Pastikan server_address dan server_port sesuai

### Lag/Delay

- Turunkan quality: `--quality 70`
- Kecilkan ukuran: `--max_size 480`
- Pastikan network stabil (untuk remote connection)

## Network Architecture

```
[Webcam] → [Python Server] → [UDP Socket] → [Godot Client] → [Display]
            - Face Detection
            - Mask Overlay
            - JPEG Encode
```

UDP Protocol:

- Header: 4 bytes (frame size, big-endian)
- Body: JPEG image data

## Performance Tips

Untuk FPS optimal:

1. **Python Server**:

   - Gunakan model yang sudah trained
   - Kurangi `--max_size` jika lag
   - Gunakan `--quality 70-80` untuk balance

2. **Network**:

   - Local: gunakan 127.0.0.1 (paling cepat)
   - LAN: gunakan IP address komputer server
   - WAN: tidak direkomendasikan (latency tinggi)

3. **Godot Client**:
   - Sudah optimal, tidak perlu tuning

## License

Mengikuti lisensi proyek utama PCD-TryOn-Mask
