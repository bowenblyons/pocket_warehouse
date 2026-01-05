

## Docker

Set up to use QEMU to translate ARM to x86:
`docker run --privileged --rm tonistiigi/binfmt --install all`

Then run docker compose to make the container and run it:
`docker compose run virtual-pi`
