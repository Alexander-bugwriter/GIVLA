# do such fix in your-libero-plus-test-root/libero/libero/env/env_wrapper.py
replace the fog function 
```python 
def fog(x, severity=1):
    c = [(0.5, 3), (1.0, 2.8), (1.5, 2.5), (2.0, 2.2),
         (2.5, 2.0), (3.0, 1.8), (3.5, 1.6), (4.0, 1.5),
         (4.5, 1.4), (5.0, 1.3)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    h, w, ch = x.shape

    noise = plasma_fractal(wibbledecay=c[1])
    noise = resize(noise, (h, w), order=1, preserve_range=True)
    noise = noise[..., np.newaxis]          # (h,w,1)
    noise = np.repeat(noise, ch, axis=-1)   # (h,w,3)

    max_val = x.max()
    x += c[0] * noise
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

```

