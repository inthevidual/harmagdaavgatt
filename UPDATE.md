# Om hon avgår

```sh
./svar.sh ja
```

Det byter **NEJ** (röd) mot **JA** (grön), committar och pushar. GitHub Pages
publicerar inom ett par minuter.

Ångra / tillbaka till nej:

```sh
./svar.sh nej
```

Bara lokalt, utan att pusha: `./svar.sh ja --no-push`

## Om sidan inte uppdateras

Cloudflare kan cacha `index.html`. Rensa i Cloudflare → *Caching* →
*Configuration* → **Purge Everything** (eller *Custom Purge* på
`https://harmagdaavgatt.nu/`).

## Övrigt

- Humöret styrs automatiskt av Polymarkets odds (marknad `1193094`,
  "Next Prime Minister of Sweden" → Magdalena Andersson) – inget att uppdatera.
- Delningsbilden (`og.jpg`) visar ett rött frågetecken och behöver inte bytas.
- Hur 3D-ansiktet och bilderna byggs: se `tools/README.md`.
