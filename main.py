import numpy as np
import re
import os
from scipy.io.wavfile import write
import sounddevice as sd
import soundfile as sf
from flask import Flask, render_template, redirect, url_for, request, jsonify ,session,send_file, flash, current_app
from dotenv import load_dotenv
#=======================================================================================================================
with open("static/assets/current.txt", "w") as f:
    f.write("")

app = Flask(__name__,static_folder="./static",template_folder="./templates")
YOUR_DOMAIN = 'http://localhost:4242'
sd.default.samplerate = 44100
load_dotenv()
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
#=======================================================================================================================
class SoundWaves:
    def __init__(self):
        self.short_beep = os.path.join(current_app.root_path, 'static', 'sound', 'beep_short.wav')
        self.long_beep =  os.path.join(current_app.root_path, 'static', 'sound', 'beep_long.wav')
        self.sound_array = []
        self.pause = np.zeros([20000, 2], dtype="float32")
        self.longer_pause = np.zeros([120000, 2], dtype="float32")

    def save_char(self, value_encoder):
        for code in value_encoder:
            if code == 1:
                data, fs = sf.read(self.short_beep)
            else:
                data, fs = sf.read(self.long_beep)

            self.sound_array.append(data)
            self.sound_array.append(self.pause)

    def save_wav_format(self) -> None:
        new = np.vstack(self.sound_array)
        int_data = np.int16(new * 32767)
        write('output.wav', 44100, int_data)

    def add_longer_pause(self) -> None:
        self.sound_array.append(self.longer_pause)

#=======================================================================================================================
def caesar_cipher(cipher,input,shift_key):
    e_or_d = cipher.lower()
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    text = input.lower()
    result = ""

    if shift_key > 25:
        shift_key = shift_key % 26

    if e_or_d == "encrypt":
        shifted = []
        for i in range(26):
            shifted.append(alphabet[(i + shift_key) % 26])
        for letter in text:
            if letter in alphabet:
                index = alphabet.index(letter)
                result += shifted[index]
            else:
                result += letter
        return result, shifted
    elif e_or_d == "decrypt":
        shifted = []
        for i in range(26):
            shifted.append(alphabet[(i + shift_key) % 26])
        for letter in text:
            if letter in alphabet:
                index = shifted.index(letter)
                result += alphabet[index]
            else:
                result += letter
        return result, shifted
def monoalphabetic_cipher(cipher,input,shifted_alphabet):
    e_or_d = cipher.lower()
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    text = input.lower()
    result = ""

    if len(shifted_alphabet) != 26:
        return "Shifted alphabet must have exactly 26 letters", ""

    shifted_list = list(shifted_alphabet.lower())

    if e_or_d == "encrypt":
        for letter in text:
            if letter in alphabet:
                index = alphabet.index(letter)
                result += shifted_list[index]
            else:
                result += letter
        print(f"The encoded text is: {result}")
        return result

    elif e_or_d == "decrypt":
        for letter in text:
            if letter in shifted_list:
                index = shifted_list.index(letter)
                result += alphabet[index]
            else:
                result += letter
        print(f"The decoded text is: {result}")
        return result
def playfair_cipher(cipher, input, keyword):
    def prepare_key(keyword):
        keyword = keyword.lower().replace("j", "i")
        seen = set()
        key_square = []
        for ch in keyword + "abcdefghiklmnopqrstuvwxyz":
            if ch not in seen and ch.isalpha():
                seen.add(ch)
                key_square.append(ch)
        matrix = [key_square[i:i + 5] for i in range(0, 25, 5)]
        return key_square, matrix

    def format_input(text):
        text = text.lower().replace("j", "i")
        formatted = []
        i = 0
        while i < len(text):
            if text[i].isalpha():
                ch1 = text[i]
                ch2 = ''
                if i + 1 < len(text) and text[i + 1].isalpha():
                    ch2 = text[i + 1]
                    if ch1 == ch2:
                        ch2 = 'x'
                        i += 1
                    else:
                        i += 2
                else:
                    ch2 = 'x'
                    i += 1
                formatted.append(ch1 + ch2)
            else:
                i += 1
        return formatted

    def get_position(ch, square):
        idx = square.index(ch)
        return idx // 5, idx % 5

    def process_digraph(ch1, ch2, square, direction):
        r1, c1 = get_position(ch1, square)
        r2, c2 = get_position(ch2, square)

        if r1 == r2:
            if direction == "encrypt":
                return square[r1 * 5 + (c1 + 1) % 5] + square[r2 * 5 + (c2 + 1) % 5]
            else:
                return square[r1 * 5 + (c1 - 1) % 5] + square[r2 * 5 + (c2 - 1) % 5]
        elif c1 == c2:
            if direction == "encrypt":
                return square[((r1 + 1) % 5) * 5 + c1] + square[((r2 + 1) % 5) * 5 + c2]
            else:
                return square[((r1 - 1) % 5) * 5 + c1] + square[((r2 - 1) % 5) * 5 + c2]
        else:
            return square[r1 * 5 + c2] + square[r2 * 5 + c1]

    square, matrix = prepare_key(keyword)
    digraphs = format_input(input)
    result = ""
    for pair in digraphs:
        ch1, ch2 = pair[0], pair[1]
        result += process_digraph(ch1, ch2, square, cipher)
    return result, matrix

def vigenere_cipher(cipher,input,keyword):
    e_or_d = cipher.lower()
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    text = input.lower()
    key = keyword.lower()
    result = ""

    key_index = 0
    for letter in text:
        if letter in alphabet:
            letter_index = alphabet.index(letter)
            key_letter = key[key_index % len(key)]
            key_shift = alphabet.index(key_letter)

            if e_or_d == "encrypt":
                shifted_index = (letter_index + key_shift) % 26
            elif e_or_d == "decrypt":
                shifted_index = (letter_index - key_shift) % 26
            result += alphabet[shifted_index]
            key_index += 1
        else:
            result += letter
    print(f"The {e_or_d}ed text is: {result}")
    return result

def vernam_cipher(cipher, input, keyword):
    alphabet = {
        'A': '01000001', 'B': '01000010', 'C': '01000011', 'D': '01000100',
        'E': '01000101', 'F': '01000110', 'G': '01000111', 'H': '01001000',
        'I': '01001001', 'J': '01001010', 'K': '01001011', 'L': '01001100',
        'M': '01001101', 'N': '01001110', 'O': '01001111', 'P': '01010000',
        'Q': '01010001', 'R': '01010010', 'S': '01010011', 'T': '01010100',
        'U': '01010101', 'V': '01010110', 'W': '01010111', 'X': '01011000',
        'Y': '01011001', 'Z': '01011010', 'a': '01100001', 'b': '01100010',
        'c': '01100011', 'd': '01100100', 'e': '01100101', 'f': '01100110',
        'g': '01100111', 'h': '01101000', 'i': '01101001', 'j': '01101010',
        'k': '01101011', 'l': '01101100', 'm': '01101101', 'n': '01101110',
        'o': '01101111', 'p': '01110000', 'q': '01110001', 'r': '01110010',
        's': '01110011', 't': '01110100', 'u': '01110101', 'v': '01110110',
        'w': '01110111', 'x': '01111000', 'y': '01111001', 'z': '01111010'
    }

    if cipher == "encrypt":
        result = []
        k = 0
        for ch in input:
            if ch in alphabet:
                input_bin = alphabet[ch]
                key_bin = alphabet[keyword[k % len(keyword)]]
                xor_bin = ""
                for i in range(8):
                    if input_bin[i] == key_bin[i]:
                        xor_bin += '0'
                    else:
                        xor_bin += '1'
                xor_value = int(xor_bin, 2)
                result.append(str(xor_value))
                k += 1
            else:
                result.append(ch)
        return " ".join(result)

    elif cipher == "decrypt":
        parts = input.split()
        if len(parts) != len(keyword):
            return "Keyword must match number of values."
        result = ""
        for i in range(len(parts)):
            xor_value = int(parts[i])
            key_bin = alphabet[keyword[i % len(keyword)]]
            key_value = int(key_bin, 2)
            original_value = xor_value ^ key_value
            result += chr(original_value)
        return result

def rail_fence_cipher(cipher, input, depth):
    input_text = input.replace('\n', '').replace(' ', '')
    length = len(input_text)
    result = ""

    if cipher == "encrypt":
        grid = [['.' for _ in range(length)] for _ in range(depth)]
        cycle = 2 * (depth - 1)

        for pos in range(length):
            row = pos % cycle
            if row >= depth:
                row = cycle - row
            grid[row][pos] = input_text[pos]

        for i in range(depth):
            for j in range(length):
                if grid[i][j] != '.':
                    result += grid[i][j]
        return result

    elif cipher == "decrypt":
        grid = [['.' for _ in range(length)] for _ in range(depth)]
        cycle = 2 * (depth - 1)
        index = 0

        for pos in range(length):
            row = pos % cycle
            if row >= depth:
                row = cycle - row
            if index < length:
                grid[row][pos] = '*'
                index += 1

        index = 0
        for i in range(depth):
            for j in range(length):
                if grid[i][j] == '*' and index < length:
                    grid[i][j] = input_text[index]
                    index += 1

        result = [''] * length
        index = 0
        for pos in range(length):
            row = pos % cycle
            if row >= depth:
                row = cycle - row
            if grid[row][pos] != '.':
                result[pos] = grid[row][pos]
                index += 1
        return ''.join(result)
def rail_fence_pattern(input_text, depth):
    input_text = input_text.replace('\n', '').replace(' ', '')
    length = len(input_text)

    grid = [['.' for _ in range(length)] for _ in range(depth)]

    cycle = 2 * (depth - 1)

    for pos in range(length):
        row = pos % cycle
        if row >= depth:
            row = cycle - row
        if pos < length:
            grid[row][pos] = input_text[pos]

    return grid

def row_transposition_cipher(cipher, input, key):
    input = input.replace(" ", "")
    key_len = len(key)
    result = ""
    matrix = []

    if cipher == "encrypt":
        i = 0
        while i < len(input):
            row = input[i:i+key_len]
            if len(row) < key_len:
                row += 'x' * (key_len - len(row))
            matrix.append(list(row))
            i += key_len

        for ch in sorted(set(key)):
            for col in range(key_len):
                if key[col] == ch:
                    for row in matrix:
                        result += row[col]

        print(f"The encrypted text is: {result}")
        print("Encryption matrix:", matrix)
        return result, matrix

    elif cipher == "decrypt":
        num_rows = len(input) // key_len
        columns = ['' for _ in range(key_len)]
        k = 0

        for ch in sorted(set(key)):
            for col in range(key_len):
                if key[col] == ch:
                    columns[col] = input[k:k+num_rows]
                    k += num_rows

        for i in range(num_rows):
            row = []
            for j in range(key_len):
                row.append(columns[j][i])
            matrix.append(row)
            result += ''.join(row)

        print(f"The decrypted text is: {result}")
        print("Decryption matrix:", matrix)
        return result, matrix

def auto_key_cipher(cipher, input, key):
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    input = input.lower()
    key = key.lower()
    result = ""

    clean_input = ''.join([ch for ch in input if ch.isalpha()])

    if cipher == "encrypt":
        full_key = (key + clean_input)[:len(clean_input)]
    elif cipher == "decrypt":
        full_key = key
    else:
        print("Error: Cipher must be 'encrypt' or 'decrypt'")
        return "", ""

    k = 0
    for i in range(len(input)):
        ch = input[i]
        if ch.isalpha():
            ch_index = alphabet.index(ch)
            key_index = alphabet.index(full_key[k])
            if cipher == "encrypt":
                new_index = (ch_index + key_index) % 26
                result += alphabet[new_index]
                k += 1
            elif cipher == "decrypt":
                new_index = (ch_index - key_index + 26) % 26
                decrypted_letter = alphabet[new_index]
                result += decrypted_letter
                full_key += decrypted_letter
                k += 1
        else:
            result += ch

    print(f"The {cipher}ed text is: {result}")
    return result

#=======================================================================================================================
#=======================================================================================================================
@app.route("/",methods=["GET","POST"])
def home():
    return render_template("choose.html")

@app.route("/method",methods = ["POST"])
def choose_method():
    cipher = request.form.get("cipher")
    if cipher == "caesar":
        return redirect(url_for("caesar"))
    elif cipher == "monoalphabetic":
        return redirect(url_for("monoalphabetic"))
    elif cipher == "playfair":
        return redirect(url_for("playfair"))
    elif cipher == "vigenere":
        return redirect(url_for("vigenere"))
    elif cipher == "vernam":
        return redirect(url_for("vernam"))
    elif cipher == "railfence":
        return redirect(url_for("rail_fence"))
    elif cipher == "rowtransposition":
        return redirect(url_for("row_transposition"))
    elif cipher == "autokey":
        return redirect(url_for("auto_key"))
    elif cipher == "morse":
        return redirect(url_for("morse"))
    else:
        return jsonify({"error": {"message":"invalid method"}}), 400

@app.route("/method/caesar", methods=["POST","GET"])
def caesar():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        print("p",previous_cipher)
        text = request.form.get("input1")
        shift = int(request.form.get("shift"))
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        result, shifted_alphabet = caesar_cipher(cipher,text,shift)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)

        ss = {
            "input_text":text,
            "output":result,
            "shift":shift,
            "shifted_alphabet":shifted_alphabet,
            "previous_cipher": previous_cipher
        }
        return render_template("caesar_page.html",**ss)
    return render_template("caesar_page.html",previous_cipher = previous_cipher)

@app.route("/method/monoalphabetic", methods=["POST","GET"])
def monoalphabetic():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        text = request.form.get("input1")
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        key = {}
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for letter in alphabet:
            key_value = request.form.get(f"key_{letter}", "").upper()
            if key_value and key_value in alphabet:
                key[letter] = key_value
        key_values = "".join(key.values())
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        print(text)
        print(is_encrypt_mode)
        print(key_values)
        result = monoalphabetic_cipher(cipher,text,key_values)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text":text,
            "output":result,
            "previous_cipher": previous_cipher
        }
        return render_template("monoalphabetic.html",**ss)
    return render_template("monoalphabetic.html",previous_cipher = previous_cipher)

@app.route("/method/playfair", methods=["POST","GET"])
def playfair():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        text = request.form.get("input1")
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        keyword = request.form.get("keyword")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        result, matrix = playfair_cipher(cipher,text,keyword)
        display_matrix = []
        for row in matrix:
            display_row = []
            for ch in row:
                if ch.lower() == 'i':
                    display_row.append("I/J")
                else:
                    display_row.append(ch.upper())
            display_matrix.append(display_row)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text":text,
            "output":result,
            "keyword":keyword,
            "generated_matrix":display_matrix,
            "previous_cipher": previous_cipher
        }
        return render_template("playfair_page.html",**ss)
    return render_template("playfair_page.html",previous_cipher = previous_cipher)

@app.route("/method/vigenere", methods=["POST","GET"])
def vigenere():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)

    if request.method == "POST":
        text = request.form.get("input1")
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        keyword = request.form.get("keyword")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        result = vigenere_cipher(cipher,text,keyword)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text":text,
            "output":result,
            "keyword":keyword,
            "previous_cipher":previous_cipher
        }
        return render_template("vigenere_page.html",**ss)
    return render_template("vigenere_page.html",previous_cipher = previous_cipher)

@app.route("/method/vernam", methods=["POST","GET"])
def vernam():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        text = request.form.get("input1")
        keyword = request.form.get("keyword")
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        result = vernam_cipher(cipher,text,keyword)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text":text,
            "output":result,
            "keyword":keyword,
            "previous_cipher": previous_cipher
        }
        return render_template("vernam_page.html",**ss)
    return render_template("vernam_page.html",previous_cipher = previous_cipher)

@app.route("/method/rail_fence", methods=["POST", "GET"])
def rail_fence():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        text = request.form.get("input1")
        depth = int(request.form.get("depth"))
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
            result = rail_fence_cipher(cipher, text, depth)
            pattern = rail_fence_pattern(text, depth)
        else:
            cipher = "decrypt"
            result = rail_fence_cipher(cipher, text, depth)
            pattern = rail_fence_pattern(result, depth)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text": text,
            "output": result,
            "depth": depth,
            "pattern":pattern,
            "previous_cipher": previous_cipher
        }
        return render_template("rail_fence_page.html", **ss)
    return render_template("rail_fence_page.html",previous_cipher = previous_cipher)

@app.route("/method/row_transposition", methods=["POST", "GET"])
def row_transposition():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        text = request.form.get("input1")
        key = str(request.form.get("key"))
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        result, matrix = row_transposition_cipher(cipher, text, key)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text": text,
            "output": result,
            "key": key,
            "matrix": matrix,
            "key_length": list(key),
            "previous_cipher": previous_cipher
        }
        return render_template("row_transposition.html", **ss)
    return render_template("row_transposition.html",previous_cipher = previous_cipher)

@app.route("/method/auto_key", methods=["POST", "GET"])
def auto_key():
    with open("static/assets/current.txt", "r") as f:
        previous_cipher = f.read().strip()
    print(previous_cipher)
    if request.method == "POST":
        text = request.form.get("input1")
        is_encrypt_mode = request.form.get("is_encrypt_mode")
        keyword = request.form.get("keyword")
        if is_encrypt_mode == "true":
            cipher = "encrypt"
        else:
            cipher = "decrypt"
        result = auto_key_cipher(cipher, text, keyword)
        with open("static/assets/current.txt", "w") as f:
            f.write(result)
        ss = {
            "input_text": text,
            "output": result,
            "keyword": keyword,
            "previous_cipher":previous_cipher,
        }
        return render_template("autokey_page.html", **ss)
    return render_template("autokey_page.html",previous_cipher = previous_cipher)


@app.route("/method/morse", methods=["GET", "POST"])
def morse():
    convert_dict = {
        "A": [1, 2], "B": [2, 1, 1, 1], "C": [2, 1, 2, 1], "D": [2, 1, 1],
        "E": [1], "F": [1, 1, 2, 1], "G": [2, 2, 1], "H": [1, 1, 1, 1],
        "I": [1, 1], "J": [1, 2, 2, 2], "K": [2, 1, 2], "L": [1, 2, 1, 1],
        "M": [2, 2], "N": [2, 1], "O": [2, 2, 2], "P": [1, 2, 2, 1],
        "Q": [2, 2, 1, 2], "R": [1, 2, 1], "S": [1, 1, 1], "T": [2],
        "U": [1, 1, 2], "V": [1, 1, 1, 2], "W": [1, 2, 2], "X": [2, 1, 1, 2],
        "Y": [2, 1, 2, 2], "Z": [2, 2, 1, 1],
        "0": [2, 2, 2, 2, 2], "1": [1, 2, 2, 2, 2], "2": [1, 1, 2, 2, 2],
        "3": [1, 1, 1, 2, 2], "4": [1, 1, 1, 1, 2], "5": [1, 1, 1, 1, 1],
        "6": [2, 1, 1, 1, 1], "7": [2, 2, 1, 1, 1], "8": [2, 2, 2, 1, 1],
        "9": [2, 2, 2, 2, 1]
    }
    input_text = ""
    audio_file = False
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip().upper()
        if input_text and re.match(r'^[A-Z0-9\s]+$', input_text):
            new_sound = SoundWaves()
            convert = list(input_text)
            for char in convert:
                char = char.upper()
                if char in convert_dict:
                    value = convert_dict[char]
                    new_sound.save_char(value)
                else:
                    new_sound.add_longer_pause()
            new_sound.save_wav_format()
            audio_file = True
        ss = {
            "input_text" : input_text,
            "audio_file" : audio_file
        }
        return render_template("morse_code_page.html", **ss)
    return render_template("morse_code_page.html")

@app.route("/morse_audio",methods = ["POST","GET"])
def serve_audio():
    return send_file("output.wav", mimetype="audio/wav")


if __name__ == '__main__':
     app.run(debug=True ,port=5000)