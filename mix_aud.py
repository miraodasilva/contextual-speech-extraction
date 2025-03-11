import numpy as np

def mix_audio_3spk(signal, noise1, noise2, snr1, snr2, pad=False):
    # if the audio is longer than the noise
    # if pad is true the noise is zero padded to match the length.
    # else play the noise in repeat for the duration of the audio
    sig_len = len(signal)
    ns1_len = len(noise1)
    ns2_len = len(noise2)
    max_len = max([sig_len, ns1_len, ns2_len])
    if not pad:
        if max_len > len(signal):
            signal = signal[np.arange(max_len) % len(signal)]
        if max_len > len(noise1):
            noise1 = noise1[np.arange(max_len) % len(noise1)]
        if max_len > len(noise2):
            noise2 = noise2[np.arange(max_len) % len(noise2)]
        
    # this is important if loading resulted in 
    # uint8 or uint16 types, because it would cause overflow
    # when squaring and calculating mean
    noise1 = noise1.astype(np.float32)
    noise2 = noise2.astype(np.float32)
    signal = signal.astype(np.float32)
    
    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise1_energy = np.mean(noise1**2)
    noise2_energy = np.mean(noise2**2)

    # calculates the gain to be applied to the noise to achieve the given SNR
    g1 = np.sqrt(10.0 ** (-snr1/10) * signal_energy / noise1_energy)
    g2 = np.sqrt(10.0 ** (-snr2/10) * signal_energy / noise2_energy)

    if pad:
        if max_len > len(signal):
            signal = np.concatenate([signal, np.zeros(max_len - len(signal))], 0)
        if max_len > len(noise1):
            noise1 = np.concatenate([noise1, np.zeros(max_len - len(noise1))], 0)
        if max_len > len(noise2):
            noise2 = np.concatenate([noise2, np.zeros(max_len - len(noise2))], 0)

    noise1 = g1 * noise1
    noise2 = g2 * noise2

    mixed_audio = signal + noise1 + noise2

    scale = 1 / np.max(np.abs(mixed_audio)) * 0.9
    mixed_audio = scale * mixed_audio
    signal = scale * signal
    noise1 = scale * noise1
    noise2 = scale * noise2
    return mixed_audio, signal, noise1, noise2

#for 2spk, normalize by peak amplitude
def mix_audio(signal, noise, snr, pad=False):
    # if the audio is longer than the noise
    # if pad is true the noise is zero padded to match the length.
    # else play the noise in repeat for the duration of the audio
    if not pad and len(signal) > len(noise):
        noise = noise[np.arange(len(signal)) % len(noise)]
    if len(signal) < len(noise):
        noise = noise[:len(signal)]
    # this is important if loading resulted in 
    # uint8 or uint16 types, because it would cause overflow
    # when squaring and calculating mean
    noise = noise.astype(np.float32)
    signal = signal.astype(np.float32)
    
    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise 
    # to achieve the given SNR
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
    
    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of 
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))

    if pad and len(signal) > len(noise):
        noise = np.concatenate([noise, np.zeros(len(signal) - len(noise))], 0)
    
    # mix the signals
    signal = a * signal
    noise = b * noise

    mixed_audio = signal + noise

    scale = 1 / np.max(np.abs(mixed_audio)) * 0.9
    mixed_audio = scale * mixed_audio
    signal = scale * signal
    noise = scale * noise
    return mixed_audio, signal, noise

if __name__ == "__main__":
    import librosa

    ## 2 spk case
    source_aud_path = './source_aud.wav'
    noise_aud_path = './noise_aud.wav'

    snr = '5'

    source_aud, sr = librosa.load(source_aud_path, sr=16000)
    noise_aud, _ = librosa.load(noise_aud_path, sr=16000)
    
    noise_aud = noise_aud / np.max(np.abs(noise_aud)) * 0.9
    source_aud = source_aud / np.max(np.abs(source_aud)) * 0.9
    assert sr == 16000
    
    snr = float(snr)
    mixed_audio, source_aud, noise_aud = mix_audio(source_aud, noise_aud, snr, pad=True)

    ## 3 spk case
    source_aud_path = './source_aud.wav'
    noise_aud_path_1 = './noise_aud_1.wav'
    noise_aud_path_2 = './noise_aud_2.wav'

    snr1 = '5'
    snr2 = '-2'

    source_aud, sr = librosa.load(source_aud_path, sr=16000)
    noise_aud_1, _ = librosa.load(noise_aud_path_1, sr=16000)
    noise_aud_2, _ = librosa.load(noise_aud_path_2, sr=16000)

    noise_aud_1 = noise_aud_1 / np.max(np.abs(noise_aud_1)) * 0.9
    noise_aud_2 = noise_aud_2 / np.max(np.abs(noise_aud_2)) * 0.9
    source_aud = source_aud / np.max(np.abs(source_aud)) * 0.9
    
    assert sr == 16000
    
    snr1 = float(snr1)
    snr2 = float(snr2)

    # mix audio at specific SNR level
    mixed_audio, source_aud, noise_aud_1, noise_aud_2 = mix_audio_3spk(source_aud, noise_aud_1, noise_aud_2, snr1, snr2, pad=True)