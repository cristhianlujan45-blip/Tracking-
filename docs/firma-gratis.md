# Quitar el aviso de SmartScreen gratis

Windows muestra «Windows protegió su PC» con los programas que no tienen una firma digital de
confianza. Un certificado de firma de código cuesta dinero cada año y un certificado «casero»
(autofirmado) **no** quita el aviso. Estas son las vías gratuitas, de la más recomendable a la menos:

## 1. Microsoft Store (gratis, recomendado)

Microsoft firma la app al publicarla: instalarla desde la Store no muestra ningún aviso y se
actualiza sola. La cuenta de desarrollador es gratuita para particulares.

1. Crea la cuenta en <https://storedeveloper.microsoft.com> (tipo **Individual**, gratis; pide
   verificar tu identidad).
2. En **Partner Center → Apps y juegos → Nueva app**, reserva el nombre «Forja3D» (u otro libre).
3. En la app reservada: **Administración del producto → Identidad del producto**. Copia:
   - `Package/Identity/Name` (p. ej. `12345TuNombre.Forja3D`)
   - `Package/Identity/Publisher` (empieza por `CN=`)
   - `Package/Properties/PublisherDisplayName`
4. En GitHub: **Settings → Secrets and variables → Actions → Variables**, crea:
   `MSIX_IDENTITY_NAME`, `MSIX_PUBLISHER` y `MSIX_PUBLISHER_DISPLAY` con esos valores.
5. La próxima compilación genera `Forja3D-x.y.z-store.msix` (en *Actions → la ejecución →
   Artifacts → Forja3D-Microsoft-Store*). Súbelo en **Partner Center → Envío → Paquetes**,
   completa la ficha (descripción, capturas, clasificación por edades) y envíalo a certificación.
   La app usa la capacidad `runFullTrust` (programa de escritorio): explica en las notas para
   certificación que es un estudio 3D que ejecuta su IA local en la GPU del usuario.

## 2. Actualizaciones desde la propia app (ya incluido)

Lo que descarga la app no lleva la marca «descargado de internet», así que **las
actualizaciones no muestran el aviso**: solo lo verás la primera vez que instalas desde el
navegador. La app comprueba la última versión publicada y, en Windows, la descarga, la instala
en silencio y se vuelve a abrir.

Para eso los instaladores deben poder descargarse sin iniciar sesión. Si el repositorio del
código es privado, crea un repositorio **público** solo para los instaladores:

1. Crea un repositorio público vacío, p. ej. `tu-usuario/Forja3D` (no hace falta subir código).
2. Crea un token: GitHub → Settings → Developer settings → Fine-grained tokens, con acceso a ese
   repositorio y permiso **Contents: Read and write**.
3. En el repositorio del código: variable `RELEASES_REPO` = `tu-usuario/Forja3D` y secreto
   `RELEASES_TOKEN` = el token.

A partir de la siguiente compilación los instaladores se publican también allí y la app se
actualiza desde ese repositorio.

## 3. SignPath Foundation (gratis para código abierto)

Si algún día publicas el código como software libre (repositorio público con licencia
OSI), <https://signpath.org> firma gratis los instaladores con un certificado de confianza.
Se solicita en su web; una vez aprobado se añade su paso de firma a `.github/workflows/desktop.yml`.

## 4. winget

Con los instaladores en un repositorio público también se puede publicar en `winget`
(`winget install Forja3D`), que instala sin pasar por el navegador. Se hace con un *pull request*
a <https://github.com/microsoft/winget-pkgs> (por ejemplo con `wingetcreate new <url del .exe>`).
