# Maintainer: Matthew Wardrop <mister.wardrop@gmail.com>
pkgname=python2-qubricks
pkgver=0.9
pkgrel=1
pkgdesc="A parameter manager that keeps track of physical (or numerical) quantities, and the relationships between them."
arch=('i686' 'x86_64')
url=""
license=('GPL')
groups=()
depends=('python2' 'python2-numpy' 'python2-sympy' 'python2-scipy' 'python2-parameters>=1.1.8' )
makedepends=()
provides=()
conflicts=()
replaces=()
backup=()
options=(!emptydirs)
install=
source=()
md5sums=()

package() {
  cd ".."
  #cd "$srcdir/$pkgname-$pkgver"
  python2 setup.py install --root="$pkgdir/" --optimize=1
}

# vim:set ts=2 sw=2 et:
