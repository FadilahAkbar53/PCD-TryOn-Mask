# 🎮 Godot Input Map Setup

## Cara Setup Keyboard Controls di Godot

Ikuti langkah-langkah ini untuk menambahkan input mapping di Godot:

### Step 1: Buka Project Settings
1. Di Godot Editor, klik **Project** → **Project Settings**
2. Pilih tab **Input Map**

### Step 2: Tambahkan Actions

Tambahkan action-action berikut satu per satu:

#### Action: `q` (Quit)
1. Ketik `q` di kolom "Add New Action"
2. Klik tombol **Add**
3. Klik tombol **+** di sebelah kanan action `q`
4. Tekan tombol **Q** di keyboard
5. Klik **OK**

#### Action: `m` (Toggle Mask)
1. Ketik `m` di kolom "Add New Action"
2. Klik **Add**
3. Klik **+** → Tekan **M** → **OK**

#### Action: `s` (Screenshot)
1. Ketik `s` di kolom "Add New Action"
2. Klik **Add**
3. Klik **+** → Tekan **S** → **OK**

#### Action: `1` sampai `7` (Switch Mask)
Ulangi untuk angka 1-7:
1. Ketik `1` di kolom "Add New Action" → **Add**
2. Klik **+** → Tekan **1** → **OK**
3. Ulangi untuk `2`, `3`, `4`, `5`, `6`, `7`

### Step 3: Verifikasi

Pastikan semua action ini sudah ada:
- ✅ `q` → Q (Physical)
- ✅ `m` → M (Physical)
- ✅ `s` → S (Physical)
- ✅ `1` → 1 (Physical)
- ✅ `2` → 2 (Physical)
- ✅ `3` → 3 (Physical)
- ✅ `4` → 4 (Physical)
- ✅ `5` → 5 (Physical)
- ✅ `6` → 6 (Physical)
- ✅ `7` → 7 (Physical)

### Step 4: Save & Test

1. Klik **Close** untuk menutup Project Settings
2. Tekan **F5** untuk run project
3. Test keyboard:
   - Tekan **M** → Mask on/off
   - Tekan **1-7** → Ganti mask
   - Tekan **S** → Screenshot
   - Tekan **Q** → Quit

## Alternative: Import Project Settings

Jika Anda sudah punya screenshot input map (seperti yang di-attach), settings sudah tersimpan di `project.godot`. Cukup:

1. Buka project di Godot
2. Input map otomatis ter-load
3. Langsung run (F5)

## Troubleshooting

### Input tidak bekerja
**Cek:**
1. Project Settings → Input Map → pastikan semua action ada
2. Window Godot harus fokus (klik window Godot)
3. Lihat console Godot untuk log command

### Command terkirim tapi tidak execute
**Cek:**
1. Python server running dengan output: `Listening for commands on port 5006`
2. Firewall tidak block port 5006
3. Lihat log di terminal Python: `Received command from Godot: ...`

## Technical Details

**Command Flow:**
```
[Godot] Input.is_action_just_pressed("m")
   ↓
[Godot] send_command("toggle_mask")
   ↓
[UDP] 127.0.0.1:5006
   ↓
[Python] check_commands() receives "toggle_mask"
   ↓
[Python] handle_command() executes
   ↓
[Python] inference.toggle_mask()
```

**Port Usage:**
- **5005**: Video stream (Python → Godot)
- **5006**: Commands (Godot → Python)

---

**Setup selesai! Test dengan menekan tombol M, 1-7, S, atau Q di Godot window.** 🎉
