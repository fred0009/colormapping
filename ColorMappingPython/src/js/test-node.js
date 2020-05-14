var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

var geometry = new THREE.BoxGeometry();
var material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
var cube = new THREE.Mesh( geometry, material );
scene.add( cube );

camera.position.z = 5;
function animate(arg) {
    tmp = {1:1, 2:5};
    if (arg == tmp[arg]) {
    var geometry2 = new THREE.BoxGeometry(1,1,6);
    var material2 = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
    var cube2 = new THREE.Mesh( geometry2, material2 );
    scene.add(cube2);
    console.log('sdfsdfsdfsdfsd')
  //  block of code to be executed if the condition is true
    }
	requestAnimationFrame( animate );
	renderer.render( scene, camera );
}
