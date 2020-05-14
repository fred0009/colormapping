colors = [{'hue':'10R', 'value':8, 'chroma':8, 'rgb':'134,152,159'}]
function doRender() {
    var camera, controls, scene, renderer;
    const WIDTH = 600;
    const HEIGHT = 600;

    const CYLINDER_WIDTH = 10;
    const OBJ_HEIGHT = 40;
    const VERT_GAP = 5;
    const HOR_GAP = 5;
    const LEAF_OFFSET = 35;
    const Y_0 = 0 - ((OBJ_HEIGHT + VERT_GAP) * 11 / 2);
    const Y_1 = Y_0 + (OBJ_HEIGHT + VERT_GAP);
    const Y_2 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 2 ;
    const Y_3 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 3 ;
    const Y_4 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 4 ;
    const Y_5 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 5 ;
    const Y_6 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 6 ;
    const Y_7 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 7 ;
    const Y_8 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 8 ;
    const Y_9 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 9 ;
    const Y_10 = Y_0 + (OBJ_HEIGHT + VERT_GAP) * 10 ;

    const Y_POS = {'0': Y_0, '1': Y_1, '2': Y_2, '3': Y_3, '4': Y_4, '5': Y_5, '6': Y_6, '7': Y_7, '8': Y_8, '9': Y_9, '10': Y_10};

    var PIVOTS = {};

    const HUES = ['5R', '10R', '5YR', '10YR', '5Y', '10Y', '5GY', '10GY', '5G', '10G', '5BG', '10BG', '5B', '10B', '5PB', '10PB', '5P', '10P', '5RP', '10RP'];

    init();
    render();

    function init() {
        scene = new THREE.Scene();
        renderer = new THREE.WebGLRenderer();
        renderer.setClearColor('#ffffff');
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(WIDTH, HEIGHT);
        var container = document.getElementById('scene_div');
        container.appendChild(renderer.domElement);
        camera = new THREE.PerspectiveCamera( 75, WIDTH / HEIGHT, 1, 5000 );



        camera.position.z = 400;
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.addEventListener('change', render);


        // world
        drawMunsell();


        var light = new THREE.DirectionalLight( 0xffffff );
        light.position.set( 1, 1, 1 );
        spotLight.castShadow = false;
        scene.add( light );
        var light = new THREE.DirectionalLight( 0xffffff );
        light.position.set( -1, -1, -1 );
        spotLight.castShadow = false;
        scene.add( light );
        var light = new THREE.AmbientLight( 0x404040 );
        spotLight.castShadow = false;
        scene.add( light );

        renderer.shadowMap.enabled = false;
        light.castShadow = false;

        window.addEventListener('resize', onWindowResize, false);
    }

    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        render();
    }

    function render() {
        renderer.render(scene, camera);
    }

    function createPivot(degrees) {
        var pivot = new THREE.Object3D();
        pivot.rotation.y = degrees * Math.PI / 180;
        return pivot;
    }

    function drawMunsell() {
        const parent = new THREE.Object3D();
        scene.add( parent );

        var degrees = 0;
        for (var i = 0; i < HUES.length; i++) {
            var pivot = createPivot(degrees);
            parent.add(pivot);
            PIVOTS[HUES[i]] = pivot;
            degrees+=18;
        }

        var arrayLength = colors.length;
        for (i = 0; i < arrayLength; i++) {
            drawColor(colors[i]);
        }
    }

    function drawColor(color) {
        const hue = color['hue'];
        const chroma = parseInt(color['chroma']);
        var value = color['value'];
        const rgb = color['rgb'];

        if (hue.startsWith('N')) {
            value = hue.substring(1);
            const cyl = createCylinder(rgb, value);
            scene.add(cyl);
        } else {
            createLeaf("rgb(" + rgb + ")", LEAF_OFFSET + ((chroma / 2 - 1) * (OBJ_HEIGHT + HOR_GAP)), Y_POS[value], PIVOTS[hue]);
        }
    }

    function createCylinder(color, value) {
        const material = new THREE.MeshLambertMaterial({color: 'rgb(' + color + ')'});
        const cylinder = new THREE.Mesh(new THREE.CylinderGeometry(CYLINDER_WIDTH, CYLINDER_WIDTH, OBJ_HEIGHT, 16), material);
        cylinder.position.z = 0;
        cylinder.position.x = 0;
        cylinder.position.y = Y_POS[value];
        return cylinder;
    }

    function createLeaf(color, positionX, positionY, pivot) {
        const material = new THREE.MeshLambertMaterial({color: color});
        const leaf = new THREE.Mesh(new THREE.BoxGeometry(40, 40, 4), material);
        leaf.position.x = positionX;
        leaf.position.y = positionY;
        pivot.add(leaf);
        return leaf;
    }
}