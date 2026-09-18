import * as THREE from "https://esm.sh/three";
import { OrbitControls } from "https://esm.sh/three/examples/jsm/controls/OrbitControls.js";
import {GUI} from "https://esm.sh/lil-gui"
import * as CMAP from "https://esm.sh/colormap";
import { LineGeometry } from 'https://esm.sh/three/addons/lines/LineGeometry.js';
import { Line2 } from 'https://esm.sh/three/addons/lines/Line2.js';
import { LineMaterial } from 'https://esm.sh/three/addons/lines/LineMaterial.js';
import { FontLoader } from 'https://esm.sh/three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'https://esm.sh/three/addons/geometries/TextGeometry.js';

const loader2 = new FontLoader();
const font = await loader2.loadAsync( 'https://esm.sh/three@0.183.0/examples/fonts/helvetiker_regular.typeface.json' );

function render({ model, el }) {
    ///////////////  Create Containers ///////////////
    let container = document.createElement("div");
    let subcontainer = document.createElement("div");
    
    container.id = 'PlotWindow';
    container.style.height = "35rem";
    container.style.width = "100%";
    subcontainer.setAttribute(
      'style',
      'position: relative; resize:neither; overflow:auto;'
    );

    container.appendChild(subcontainer);
    el.appendChild(container);

    ////////////// StopWatch Class //////////////////
    
    class StopWatch extends THREE.Timer {
        constructor(paused = false){
            super();
            this.time = 0.0;
            this.startTime = 0.0;
            this.stopTime = 0.0;
            this.totalPause = 0.0;
            this.totalRun = 0.0;
            this.lastResetTime = 0.0;
            this.paused = paused;
        }
        
        reset(){
            this.lastResetTime = super.getElapsed();
            this.totalPause = 0.0;
            this.totalRun = 0.0;
            this.startTime = this.lastResetTime;
            this.stopTime = this.lastResetTime;
        }
        pause(){
            if(this.paused){
                // already paused, start again
                this.startTime = super.getElapsed();
                this.totalPause += this.startTime - this.stopTime;
                this.paused = false;
            }
            else{ // currently running, stop it
                this.stopTime = super.getElapsed();
                this.totalRun += this.stopTime - this.startTime;
                this.paused = true;
            }
        }
        run(){
            super.update();
            if(this.paused){
                this.time = this.totalRun;
            }
            else{
                this.time = super.getElapsed() - this.totalPause - this.lastResetTime;
            }
        }
    }
    
    const stopwatch = new StopWatch();

    //////////// Get Data from Model //////////////
    
    let trig_bytes = new Uint32Array(
        model.get("_trig_bytes").buffer,
    );
    let base_bytes = new Float32Array(
        model.get("_base_bytes").buffer
    );
    let guided_bytes = new Float32Array(
        model.get("_guided_bytes").buffer
    );
    let propagating_bytes = new Float32Array(
        model.get("_prop_bytes").buffer
    );
    let evanescent_bytes = new Float32Array(
        model.get("_ev_bytes").buffer
    );
    let f0s = new Float32Array(
        model.get("_f0_bytes").buffer
    );
    let x0s = new Float32Array(
        model.get("_x0_bytes").buffer
    );
    let zlims = new Float32Array(
        model.get("_zlim_bytes").buffer
    );

    ////////// Get Data Limits /////////////////
    
    const xmin = Math.min(...x0s);
    const xmax = Math.max(...x0s);

        
    const ymax = Math.max(...f0s);
    const ymin = -Math.max(...f0s);
        
    const zmin = zlims[0]
    const zmax = zlims[1]
    const ztranslation = -(zmax - zmin)/2;

    //////////// Get Colormaps //////////////////
    
    let viridis = CMAP.default({
        colormap: 'viridis',
        nshades: 256,
        format: 'float',
        alpha: 1
    })
    
    let magma = CMAP.default({
        colormap: 'magma',
        nshades: 256,
        format: 'float',
        alpha: 1
    })
    
    let rainbow = CMAP.default({
        colormap: 'rainbow',
        nshades: 256,
        format: 'float',
        alpha: 1
    })

    let bone = CMAP.default({
        colormap: 'bone',
        nshades: 256,
        format: 'float',
        alpha: 1
    })
    
    let copper = CMAP.default({
        colormap: 'copper',
        nshades: 256,
        format: 'float',
        alpha: 1
    })

    var viridis_lut = [];
    for (let n=0; n < 256; n++) {
        viridis_lut.push(new THREE.Vector3(viridis[n][0], viridis[n][1], viridis[n][2]));
    }
    
    var magma_lut = [];
    for (let n=0; n < 256; n++) {
        magma_lut.push(new THREE.Vector3(magma[n][0], magma[n][1], magma[n][2]));
    }
    
    var rainbow_lut = [];
    for (let n=0; n < 256; n++) {
        rainbow_lut.push(new THREE.Vector3(rainbow[n][0], rainbow[n][1], rainbow[n][2]));
    }

    var bone_lut = [];
    for (let n=0; n < 256; n++) {
        bone_lut.push(new THREE.Vector3(bone[n][0], bone[n][1], bone[n][2]));
    }
    
    var copper_lut = [];
    for (let n=0; n < 256; n++) {
        copper_lut.push(new THREE.Vector3(copper[n][0], copper[n][1], copper[n][2]));
    }
    
    const luts = {
      'viridis': viridis_lut,
      'magma': magma_lut,
      'bone': bone_lut,
      'copper': copper_lut,
      'rainbow': rainbow_lut,
    }

    ///////////  Scene, Camera, Renderer, Controls /////////
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color( 'black' );
    
    let ratio = container.clientWidth / container.clientHeight
    const camera = new THREE.PerspectiveCamera(50, ratio,
                                               0.1, 2000);
    camera.position.set( 0, 6*ymax, 1.85*ztranslation);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    subcontainer.appendChild(renderer.domElement);
    
    const controls = new OrbitControls(camera, renderer.domElement);

    //////////////  Add AxesHelper with Labels //////////////
    
    const axesLength = 1.25;
    const axesHelper = new THREE.AxesHelper( axesLength );
    axesHelper.translateX(1.08*xmin);
    axesHelper.translateZ(1.08*ztranslation);

    scene.add( axesHelper );

    const textMaterial = new THREE.MeshBasicMaterial({
        // color: new THREE.Color().setRGB(1, 0.6, 0),
        color: 'silver',
        // roughness: 1,
    });
    
    const fontsize = .25;
    const fontdepth= .075;
    
    const textGeometryX = new TextGeometry('X', {
        font: font,
        size: fontsize,
        depth: fontdepth,
    });
    
    const textGeometryY = new TextGeometry('Y', {
        font: font,
        size: fontsize,
        depth: fontdepth,
    });
    const textGeometryZ = new TextGeometry('Z', {
        font: font,
        size: fontsize,
        depth: fontdepth,
    });
    
    const textX = new THREE.Mesh(textGeometryX, textMaterial);
    const textY = new THREE.Mesh(textGeometryY, textMaterial);
    const textZ = new THREE.Mesh(textGeometryZ, textMaterial);
    
    textX.position.x = axesHelper.position.x + 1.05*axesLength;
    textX.position.y = axesHelper.position.y - .5*fontsize;
    textX.position.z = axesHelper.position.z - .5*fontdepth;
    scene.add(textX);
    
    textY.position.x = axesHelper.position.x - .5*fontsize;
    textY.position.y = axesHelper.position.y + 1.05*axesLength;
    textY.position.z = axesHelper.position.z - .5*fontdepth;
    scene.add(textY);

    textZ.rotateY(Math.PI/2);
    textZ.position.x = axesHelper.position.x - .5*fontdepth ;
    textZ.position.y = axesHelper.position.y - .5*fontsize;
    textZ.position.z = axesHelper.position.z + 1.05*axesLength + .5*fontsize;
    scene.add(textZ);
    
// });

    //////////////// Main Mesh  ///////////////////////
    
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex( new THREE.Uint32BufferAttribute(trig_bytes, 1));
    geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( base_bytes, 3 ) );
    geometry.setAttribute( 'guided', new THREE.Float32BufferAttribute( guided_bytes, 2 ) );
    geometry.setAttribute( 'propagating', new THREE.Float32BufferAttribute( propagating_bytes, 2 ) );
    geometry.setAttribute( 'evanescent', new THREE.Float32BufferAttribute( evanescent_bytes, 2 ) );
    
    geometry.computeVertexNormals();

    const uniforms = {	
        time: {type: 'f', value: stopwatch.time},
        speed: {value: 2.5},
        scale: {value: 1.0},
        guided_on: {value:true},
        evanescent_on: {value:true},
        propagating_on: {value:true},
        vLut: {type: "v3v", value: luts['viridis']},
        colormax: {type: 'f', value:ymax},
        colormin: {type: 'f', value:ymin},
        colorscale: {value:1.0},
    }

    const big_uniforms = THREE.UniformsUtils.merge( [ THREE.UniformsLib[ 'lights' ], THREE.UniformsLib[ 'fog' ], uniforms ] );
    // big_uniforms['fogDensity'] = .5;
    // big_uniforms.update;
    
const material_shader = new THREE.ShaderMaterial( {
    uniforms: big_uniforms,
    vertexShader: `
        attribute vec2 evanescent;
        attribute vec2 propagating;
        attribute vec2 guided;
        attribute vec2 height;

        uniform bool evanescent_on;
        uniform bool propagating_on;
        uniform bool guided_on;
        
        uniform float speed;
        uniform float scale;
        uniform float colormax;
        uniform float colormin;
        uniform float colorscale;
        uniform float time;
        uniform int n_colors;
        uniform vec3 vLut[256];
        varying vec3 vColor;


        void main(){
            vec4 result;
            float Real;
            float Imag;
            if (guided_on == true){
              Real += guided.x;
              Imag += guided.y;
            }
            if (evanescent_on == true){
              Real += evanescent.x;
              Imag += evanescent.y;
            }
            if (propagating_on == true){
              Real += propagating.x;
              Imag += propagating.y;
            }
            
            float ypos = (cos(speed*time)*Real + sin(speed*time)*Imag);
            
            result = vec4( position.x, scale * ypos, position.z, 1.0 );
            int index = int(floor(clamp(255.0/colorscale*(ypos-colorscale*colormin)/(colormax - colormin), 0.0, 255.0)));
            vColor = vLut[index];
            gl_Position = projectionMatrix * modelViewMatrix * result;
        }`,
    fragmentShader: `
    varying vec3 vColor;
    void main(){
      gl_FragColor = vec4(vColor,1.0);
    }`,
    side: THREE.DoubleSide,
    wireframe: false,
    lights : true,
    fog: true,
} );
    
    const mesh = new THREE.Mesh( geometry, material_shader );
    mesh.material.transparent = false;
    mesh.visible = true;
    scene.add( mesh );
    mesh.translateZ(ztranslation);

    // const shinymat = new THREE.MeshPhongMaterial(material_shader);
    // console.log(shinymat);

    /////////////  Border Lines ////////////////////

    const border_points_front = [];
    border_points_front.push( xmin, 0.0, ztranslation );
    border_points_front.push( xmax, 0.0, ztranslation );

    const border_points_left = [];
    border_points_left.push( xmin, 0.0, ztranslation );
    border_points_left.push( xmin, 0.0, ztranslation + (zmax-zmin) );
    
    const border_line_front_geo = new LineGeometry();
    border_line_front_geo.setPositions( border_points_front);

    const border_line_left_geo = new LineGeometry();
    border_line_left_geo.setPositions( border_points_left);
    
    let border_line_material = new LineMaterial( {
    
        color: new THREE.Color().setRGB (0.20784313725490197, 0.5176470588235295, 0.8941176470588236),
        // color: 'silver',
        linewidth: 1.5, // in pixels
        dashed: false
    
    } );
    
    const border_line_front = new Line2( border_line_front_geo, border_line_material );
    border_line_front.computeLineDistances();
    border_line_front.scale.set( 1, 1, 1 );
    scene.add( border_line_front );

    const border_line_back = border_line_front.clone();
    border_line_back.computeLineDistances();
    border_line_back.scale.set( 1, 1, 1 );
    border_line_back.translateZ(zmax-zmin);
    scene.add( border_line_back);

    const border_line_left = new Line2( border_line_left_geo, border_line_material );
    border_line_left.computeLineDistances();
    border_line_left.scale.set( 1, 1, 1 );
    scene.add( border_line_left );

    const border_line_right = border_line_left.clone();
    border_line_right.computeLineDistances();
    border_line_right.scale.set( 1, 1, 1 );
    border_line_right.translateX(xmax-xmin);
    scene.add( border_line_right );
    
    /////////////  Interface Lines ////////////////////
    
    const interface_points = [];
    interface_points.push( 1.0, 0.0, ztranslation );
    interface_points.push( 1.0, 0.0, zmax-zmin+ ztranslation );
    
    const interface_line_geo = new LineGeometry();
    interface_line_geo.setPositions( interface_points);
    
    let interface_line_material = new LineMaterial( {
    
        color: new THREE.Color('silver'),
        // color: new THREE.Color().setRGB (0.20784313725490197, 0.5176470588235295, 0.8941176470588236),
        linewidth: 1.25, // in pixels
        dashed: true,
        dashSize:.1,
        gapSize:.1,
    } );
    
    const interface_line = new Line2( interface_line_geo, interface_line_material );
    interface_line.computeLineDistances();
    interface_line.scale.set( 1, 1, 1 );
    scene.add( interface_line );
    
    const interface_line2 = interface_line.clone();
    interface_line2.computeLineDistances();
    interface_line2.scale.set( 1, 1, 1 );
    interface_line2.translateX(-2);
    scene.add( interface_line2 );

    
    /////////// Input Field Line //////////////////
    const f0_points = [];
    
    for ( let i = 0; i < f0s.length; i ++ ) {
        const x = x0s[i];
        const y = f0s[i];
        const z = ztranslation;
        f0_points.push( x, y, z );
    
    }
    
    const geometry2 = new LineGeometry();
    geometry2.setPositions( f0_points );
        
    let matLine = new LineMaterial( {
        color: new THREE.Color().setRGB (0.20784313725490197, 0.5176470588235295, 0.8941176470588236),
        linewidth: 2, // in pixels
        dashed: false
    } );
    
    matLine.uniforms.scale = big_uniforms.scale;
    matLine.uniforms.time = big_uniforms.time;
    matLine.uniforms.speed = big_uniforms.speed;
    
    matLine.onBeforeCompile = function ( shader ) {
    
      shader.vertexShader = 'uniform float scale;\n' + shader.vertexShader;
      shader.vertexShader = 'uniform float time;\n' + shader.vertexShader;
      shader.vertexShader = 'uniform float speed;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        'vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );',
        'vec4 start = modelViewMatrix * vec4( vec3(instanceStart.x,scale*cos(speed*time)*instanceStart.y,instanceStart.z), 1.0 );'
      ).replace(
        'vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );',
        'vec4 end = modelViewMatrix * vec4( vec3(instanceEnd.x,scale*cos(speed*time)*instanceEnd.y,instanceEnd.z), 1.0 );'
      );
    
    matLine.userData.shader = shader;
    
    };
    
    const line = new Line2( geometry2, matLine );
    line.computeLineDistances();
    line.scale.set( 1, 1, 1 );
    scene.add( line );


    /////////////////////// GUI //////////////////////////////////

    /////////////////// Set Up //////////////////////////////////
    
    const gui_container = document.createElement('div');
    gui_container.id = 'gui_container';
    gui_container.setAttribute(
      'style',
      'position: absolute; z-index: 2; display:block; right: 15px; top: 0px'
    );
    
    subcontainer.appendChild(gui_container);

    const gui = new GUI({container:gui_container});

    /////////////////////// View Folder ///////////////////////////
    
    const viewFolder = gui.addFolder('Camera');
    viewFolder.open();
    
    const viewParams = {
        
      reset_controls: function(){controls.reset()},
        
      full_screen: function (){
        const fullscreenElement =
            document.fullscreenElement || document.webkitFullscreenElement;
        if (!fullscreenElement) {
            if (window.document.documentElement.requestFullscreen) {
                container.requestFullscreen();
            } else if (canvas.webkitRequestFullscreen) {
                container.webkitRequestFullscreen();
            }
        } else if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        }
    }};
    
    viewFolder.add(viewParams, 'full_screen').name('Full Screen')
    viewFolder.add(viewParams, 'reset_controls').name('Reset View')

    ////////////// Animation Folder ////////////////
    
    const animationFolder = gui.addFolder('Animation');
    animationFolder.open();
    
    const animationParams = {
      reset_time: function(){
        stopwatch.reset()
      },
      play_pause: function(){
        stopwatch.pause()
      }
      // pause: stopwatch.paused,
    }
    
    animationFolder
      .add(animationParams, 'play_pause')
      .listen()
      .name('Play / Pause')
    
    // animationFolder
    //   .add(animationParams, 'pause')
    //   .listen()
    //   .name('pause')
    //   .onChange(function(){
    //     stopwatch.pause();
    //     console.log(stopwatch.paused);
    //   })
    
    animationFolder
      .add(animationParams, 'reset_time')
      .listen()
      .name('Restart')
    
    animationFolder.add(mesh.material.uniforms.speed, 'value', 0.0, 10.0).name('Speed');
    
    animationFolder.close();

    ////////////////// Components Folder /////////////////////
    
    const componentsFolder = gui.addFolder('Components');
    componentsFolder.open();
    const componentsParams = {
      guided_on: true,
      propagating_on: true,
      evanescent_on: true,
    };
    
    componentsFolder
      .add(componentsParams, 'guided_on')
      .name('Guided')
      .onChange((value) =>{mesh.material.uniforms.guided_on.value = value;
      });
    
    componentsFolder
      .add(componentsParams, 'propagating_on')
      .name('Propagating')
      .onChange((value) =>{mesh.material.uniforms.propagating_on.value = value;
      });
    
    componentsFolder
      .add(componentsParams, 'evanescent_on')
      .name('Evanescent')
      .onChange((value) =>{mesh.material.uniforms.evanescent_on.value = value;
      });
    
    // componentsFolder.close();

    /////////////////  Appearence Folder //////////////////////////
    
    const appearenceFolder = gui.addFolder('Appearence');
    appearenceFolder.open();
    
    const colormapList = {
      'viridis': 'viridis',
      'magma': 'magma',
      'bone':'bone',
      'copper':'copper',
      'rainbow':'rainbow',
    }
    
    const appearenceParams = {
      colormaps: colormapList['viridis'],
    };

    //////////// Subfolder : Main mesh appearence ////////////////
    
    const meshAppearenceFolder = appearenceFolder.addFolder('Field');
    meshAppearenceFolder.open();
    meshAppearenceFolder.add(mesh.material.uniforms.scale, 'value', 0.0, 2.0).name('scale');
    
    meshAppearenceFolder
      .add(appearenceParams, 'colormaps', colormapList)
      .name('colormap')
      .onChange((value) =>{mesh.material.uniforms.vLut.value = luts[value];
      });
    
    // meshAppearenceFolder.add(uniforms.colormin, 'value', ymin, 0)
    //   .listen()
    //   .name('colormin');
    // meshAppearenceFolder.add(uniforms.colormax, 'value', 0, ymax)
    //   .listen()
    //   .name('colormax');
    
    meshAppearenceFolder.add(mesh.material.uniforms.colorscale, 'value', 0.0, 2.0).name('color threshold');
    
    meshAppearenceFolder.add(mesh.material, 'wireframe');
    
    appearenceFolder.close();

    ///////////// Subfolder : Input Line Appearence ///////////////
    
    const inputFieldFolder = appearenceFolder.addFolder('Input Line');
    inputFieldFolder.open();
    
    const inputFieldParams = {
      width: 1.5,
      color: [0.20784313725490197, 0.5176470588235295, 0.8941176470588236],
    };
    
    inputFieldFolder
      .add( inputFieldParams, 'width', 0, 5 )
      .name('line width')
      .onChange( function ( value ) {
    
      matLine.linewidth = value;
    
    } );
    
    inputFieldFolder
      .addColor( inputFieldParams, 'color')
      .name('line color')
      .onChange(function (value) {
        const color = new THREE.Color().setRGB(value[0], value[1], value[2]);
        line2.material.color = color;
    });

    inputFieldFolder.close();
    // gui.close();


//////////////////// Event Listeners //////////////////
    
    window.addEventListener( 'resize', onWindowResize );

    function onWindowResize() {
    
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
    
        renderer.setSize( container.clientWidth, container.clientHeight );
    
    }
    
    renderer.domElement.addEventListener( 'click', onMouseClick);
    renderer.domElement.addEventListener( 'mousemove', onMouseMove);
    renderer.domElement.addEventListener( 'mousedown', onMouseDown);
    renderer.domElement.addEventListener( 'mouseup', onMouseUp);

    var mousedown = false;
    var mouseup = true;
    var dragging = false;
    
    function onMouseClick(){
      if(!dragging){
        stopwatch.pause();
      }
    }
    
    function onMouseMove(){
      if(mousedown){dragging=true}
      else{dragging=false}
    }
    
    function onMouseDown(){
      mousedown = true;
      mouseup = false;
    }
    
    function onMouseUp(){
      mousedown = false;
      mouseup = true;
    }
    
    // const onresize = (dom_elem, callback) => {
    //   const resizeObserver = new ResizeObserver(() => callback() );
    //   resizeObserver.observe(dom_elem);
    // };
    
    // function onDivResize() {
    //     camera.aspect = subcontainer.clientWidth / subcontainer.clientHeight;
    //     camera.updateProjectionMatrix();
    
    //     renderer.setSize( subcontainer.clientWidth, subcontainer.clientHeight );}
    
    
    // onresize(subcontainer, onDivResize);
    
    ///////////// Animate/Render Function /////////////////
    
    var requestID;
    
    function animate() {
        requestID = requestAnimationFrame(animate);
        controls.update();
        stopwatch.run();
        big_uniforms.time.value = stopwatch.time;
        big_uniforms.update;
        renderer.render(scene, camera);
    }

    //////// Call function to render and animate ///////
    
    animate();
}

export default { render };