import * as THREE from "https://esm.sh/three";
import { OrbitControls } from "https://esm.sh/three/examples/jsm/controls/OrbitControls.js";
import {GUI} from "https://esm.sh/lil-gui"
import * as CMAP from "https://esm.sh/colormap";
import { LineGeometry } from 'https://esm.sh/three/addons/lines/LineGeometry.js';
import { Line2 } from 'https://esm.sh/three/addons/lines/Line2.js';
import { LineMaterial } from 'https://esm.sh/three/addons/lines/LineMaterial.js';
import { FontLoader } from 'https://esm.sh/three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'https://esm.sh/three/addons/geometries/TextGeometry.js';
// import vert from './vert.js';

const loader2 = new FontLoader();
const font = await loader2.loadAsync( 'https://esm.sh/three@0.183.0/examples/fonts/helvetiker_regular.typeface.json' );

function render({ model, el }) {

    //#region StopWatch
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
    //#endregion
    
    //#region Import Data and Set Data Limits

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
    let y0s = new Float32Array(
        model.get("_y0_bytes").buffer
    );
    let Rhos = new Float32Array(
        model.get("_rho_bytes").buffer
    );

    ////////// Get Data Limits /////////////////
    
    const xmin = Math.min(...x0s);
    const xmax = Math.max(...x0s);
        
    const zmax = Math.max(...f0s);
    const zmin = -Math.max(...f0s);
        
    const ymin = Math.min(...y0s);
    const ymax = Math.max(...y0s);
    
    const ytranslation = -(ymax - ymin)/2.0;

    //#endregion

    //#region Get Colormaps and set LUT array

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

    //#endregion

    //#region General Setup: Divs, Scene, Camera, Renderer and Controls

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

    ///////////  Scene, Camera, Renderer, Controls /////////
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color( 'black' );
    
    let ratio = container.clientWidth / container.clientHeight
    const camera = new THREE.PerspectiveCamera(50, ratio,
                                               0.1, 2000);
    camera.position.set( 0, 2.25*ytranslation, .4*(xmax-xmin));
    camera.up.set( 0, 0, 1 );
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    subcontainer.appendChild(renderer.domElement);
    
    const controls = new OrbitControls(camera, renderer.domElement);

    //#endregion

    //#region Axis Helper

    //////////////  Add AxesHelper with Labels //////////////
    
    const axesLength = 1.25;
    const axesHelper = new THREE.AxesHelper( axesLength );
    axesHelper.translateX(1.08*xmin);
    axesHelper.translateY(1.08*ytranslation);

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

    textX.rotateX(Math.PI/2);

    textX.position.x = axesHelper.position.x + 1.05*axesLength;
    textX.position.y = axesHelper.position.y + .5*fontdepth;
    textX.position.z = axesHelper.position.z - .5*fontsize;
    scene.add(textX);

    textY.rotateX(Math.PI/2);
    textY.rotateY(Math.PI/2);
    textY.position.x = axesHelper.position.x - .5*fontdepth;
    textY.position.y = axesHelper.position.y + 1.05*axesLength;
    textY.position.z = axesHelper.position.z - .5*fontsize;
    scene.add(textY);

    textZ.rotateX(Math.PI/2);
    textZ.position.x = axesHelper.position.x - .5*fontsize ;
    textZ.position.y = axesHelper.position.y + .5*fontdepth;
    textZ.position.z = axesHelper.position.z + 1.05*axesLength + .5*fontsize;
    scene.add(textZ);

    //#endregion

    //#region Vertex and Fragment Shaders
    const vertShader = `
        attribute vec2 evanescent;
        attribute vec2 propagating;
        attribute vec2 guided;
        attribute vec2 height;

        uniform bool evanescent_on;
        uniform bool propagating_on;
        uniform bool guided_on;
        
        uniform float speed;
        uniform float scale;
        uniform float gridlinescale;
        uniform float colormax;
        uniform float colormin;
        uniform float colorscale;
        uniform float time;
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
            
            float zpos = (cos(speed*time)*Real + sin(speed*time)*Imag);
            
            result = vec4( position.x, position.y, scale * zpos, 1.0 );
            int index = int(floor(clamp(255.0/colorscale*(zpos-colorscale*colormin)/(colormax - colormin), 0.0, 255.0)));
            vColor = vLut[index];
            gl_Position = projectionMatrix * modelViewMatrix * result;
        }`;

    const fragShader =  `
    varying vec3 vColor;
    void main(){
      gl_FragColor = vec4(vColor,1.0);
    }`;

    //#endregion

    //#region Uniforms

    const uniforms = {	
        time: {value: stopwatch.time},
        speed: {value: 2.5},
        scale: {value: 1.0},
        guided_on: {value:true},
        evanescent_on: {value:true},
        propagating_on: {value:true},
        vLut: {value: luts['copper']},
        colormax: {value:zmax},
        colormin: {value:zmin},
        colorscale: {value:1.0},
        gridlinescale: {value:.7},
        shift: {value: 1.0},
    }

    // const big_uniforms = THREE.UniformsUtils.merge( [ THREE.UniformsLib[ 'lights' ], THREE.UniformsLib[ 'fog' ], uniforms ] );
    // big_uniforms['fogDensity'] = .5;
    // big_uniforms.update;

    //#endregion

    //#region Main Mesh
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex( new THREE.Uint32BufferAttribute(trig_bytes, 1));
    geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( base_bytes, 3 ) );
    geometry.setAttribute( 'guided', new THREE.Float32BufferAttribute( guided_bytes, 2 ) );
    geometry.setAttribute( 'propagating', new THREE.Float32BufferAttribute( propagating_bytes, 2 ) );
    geometry.setAttribute( 'evanescent', new THREE.Float32BufferAttribute( evanescent_bytes, 2 ) );

    geometry.computeVertexNormals();
    
    const material_shader = new THREE.ShaderMaterial( {
        uniforms: uniforms,
        vertexShader: vertShader,
        fragmentShader: fragShader,
        side: THREE.DoubleSide,
        wireframe: false,
        // lights : true,  // Do not uncomment lights and fog without merging uniforms into big_uniforms and changing things where necessary
        // fog: true,
    } );

    const mesh = new THREE.Mesh( geometry, material_shader );
    // mesh.material.transparent = false;
    mesh.visible = true;
    scene.add( mesh );
    mesh.translateY(ytranslation);
    
    //#endregion

    //#region Wireframe Overlay
    const wiremat = material_shader.clone();
    wiremat.fragmentShader =  `
        uniform float gridlinescale;
        varying vec3 vColor;
        void main(){
            gl_FragColor = vec4(gridlinescale*vColor, 1.0);
        }`;

    wiremat.wireframe = true;
    wiremat.uniforms = uniforms;
    const wiremesh = new THREE.Mesh( geometry, wiremat );
    scene.add( wiremesh );
    wiremesh.visible = false;
    wiremesh.translateY(ytranslation);
    
    //#endregion

    //#region Gridlines
    
    //#region Gridlines Setup

    let Ndata_per_point = 2;  
    let y0step = 2;
    let x0step = 2;

    let NTVgridlines = Math.floor((y0s.length - 1) / y0step) + 1;
    let NLgridlines = Math.floor((x0s.length - 1) / x0step) + 1;

    // Function to set complex valued height attributes correctly for LineGeometry
    function setHeights( array, geometry, name ) {
        // converts [ z1r, z1i, z2r, z2i,... ] to pairs format [z1r, z1i, z2r, z2i, z2r, z2i, ...]
        // and sets as attributes name+Start and name+End
        const length = array.length - 2;
        const data = new Float32Array( 2 * length );
    
        for ( let i = 0; i < length; i+=2 ) {
            data[ 2 * i ] = array[ i ];
            data[ 2 * i + 1 ] = array[ i + 1 ];
    
            data[ 2 * i + 2 ] = array[ i + 2];
            data[ 2 * i + 3 ] = array[ i + 3 ];
            }
        const instanceBuffer = new THREE.InstancedInterleavedBuffer( data, 4, 1 ); // xyz, xyz
        geometry.setAttribute( name+'Start', new THREE.InterleavedBufferAttribute( instanceBuffer, 2, 0 ) ); // xyz
        geometry.setAttribute( name+'End', new THREE.InterleavedBufferAttribute( instanceBuffer, 2, 2 ) ); // xyz
    }

    let gridline_material = new LineMaterial( {
        linewidth: 1.15,
        vertexColors: true,
    } );

    gridline_material.uniforms.time = uniforms.time;
    gridline_material.uniforms.scale = uniforms.scale;
    gridline_material.uniforms.speed = uniforms.speed;
    gridline_material.uniforms.guided_on = uniforms.guided_on;
    gridline_material.uniforms.propagating_on = uniforms.propagating_on;
    gridline_material.uniforms.evanescent_on = uniforms.evanescent_on;
    gridline_material.uniforms.vLut = uniforms.vLut;
    gridline_material.uniforms.gridlinescale = uniforms.gridlinescale;
    gridline_material.uniforms.colorscale = uniforms.colorscale;
    gridline_material.uniforms.colormin = uniforms.colormin;
    gridline_material.uniforms.colormax = uniforms.colormax;

    gridline_material.onBeforeCompile = function ( shader ) {
        shader.vertexShader = 'uniform float time;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform float scale;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform float speed;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform float colorscale;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform float colormin;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform float colormax;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform float gridlinescale;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform bool guided_on;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform bool propagating_on;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform bool evanescent_on;\n' + shader.vertexShader;
        shader.vertexShader = 'uniform vec3 vLut[256];\n' + shader.vertexShader;
        shader.vertexShader = 'attribute vec2 guidedStart;\n' + shader.vertexShader;
        shader.vertexShader = 'attribute vec2 guidedEnd;\n' + shader.vertexShader;
        shader.vertexShader = 'attribute vec2 propagatingStart;\n' + shader.vertexShader;
        shader.vertexShader = 'attribute vec2 propagatingEnd;\n' + shader.vertexShader;
        shader.vertexShader = 'attribute vec2 evanescentStart;\n' + shader.vertexShader;
        shader.vertexShader = 'attribute vec2 evanescentEnd;\n' + shader.vertexShader;
        shader.vertexShader = shader.vertexShader.replace(`		void main() {

			#ifdef USE_COLOR

				vColor.xyz = ( position.y < 0.5 ) ? instanceColorStart : instanceColorEnd;

			#endif`,
            `		void main() {
                            float RealStart;
                            float ImagStart;
                            float RealEnd;
                            float ImagEnd;
                            float resultStart;
                            float resultEnd;

                            if (guided_on == true){
                                RealStart += guidedStart.x;
                                ImagStart += guidedStart.y;
                                RealEnd += guidedEnd.x;
                                ImagEnd += guidedEnd.y;
                                }

                            if (propagating_on == true){
                                RealStart += propagatingStart.x;
                                ImagStart += propagatingStart.y;
                                RealEnd += propagatingEnd.x;
                                ImagEnd += propagatingEnd.y;
                                }

                            if (evanescent_on == true){
                                RealStart += evanescentStart.x;
                                ImagStart += evanescentStart.y;
                                RealEnd += evanescentEnd.x;
                                ImagEnd += evanescentEnd.y;
                                }

                            resultStart = cos(speed * time) * RealStart + sin(speed * time) * ImagStart;
                            resultEnd = cos(speed * time) * RealEnd + sin(speed * time) * ImagEnd;

                            #ifdef USE_COLOR
                            int indexStart = int(floor(clamp(255.0/colorscale*(resultStart-colorscale*colormin)/(colormax - colormin), 0.0, 255.0)));
                            int indexEnd = int(floor(clamp(255.0/colorscale*(resultEnd-colorscale*colormin)/(colormax - colormin), 0.0, 255.0)));
                            
                            vColor.xyz = ( position.y < 0.5 ) ? gridlinescale * vLut[indexStart] : gridlinescale * vLut[indexEnd];
                            #endif
                        \n`
        );
        shader.vertexShader = shader.vertexShader.replace(
            'vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );',
            'vec4 start = modelViewMatrix * vec4( vec3(instanceStart.x, instanceStart.y, scale * resultStart), 1.0 );'
        ).replace(
            'vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );',
            'vec4 end = modelViewMatrix * vec4( vec3(instanceEnd.x, instanceEnd.y, scale * resultEnd), 1.0 );'
        );
    gridline_material.userData.shader = shader;
    // console.log(shader.vertexShader);
    };

    // Set base gridline positions, translate resultant line later by ybase/xbase
    const TVGLBasePositions = new Float32Array(3 * x0s.length);
    const LGLBasePositions = new Float32Array(3 * y0s.length);

    for (let i = 0; i < x0s.length; i++){
        TVGLBasePositions[3 * i] = x0s[i];
        TVGLBasePositions[3 * i + 1] = 0;
        TVGLBasePositions[3 * i + 2] = 0;
    };

    for (let i = 0; i < y0s.length; i++){
        LGLBasePositions[3 * i] = 0;
        LGLBasePositions[3 * i + 1] = y0s[i];
        LGLBasePositions[3 * i + 2] = 0;
    };

    //#endregion

    //#region Transverse Gridlines

    const TVGridlines = [];

    for (let i = 0; i < NTVgridlines; i++){
        let ybase = y0s[i * y0step];

        let current_guided = guided_bytes.slice(y0step * i * Ndata_per_point * x0s.length,
        (y0step * i + 1) * Ndata_per_point * x0s.length);

        let current_prop = propagating_bytes.slice(y0step * i * Ndata_per_point * x0s.length,
        (y0step * i + 1) * Ndata_per_point * x0s.length);

        let current_ev = evanescent_bytes.slice(y0step * i * Ndata_per_point * x0s.length,
        (y0step * i + 1) * Ndata_per_point * x0s.length);

        const temp_geometry = new LineGeometry();
        temp_geometry.setPositions( TVGLBasePositions );
        setHeights(current_guided, temp_geometry, 'guided');
        setHeights(current_prop, temp_geometry, 'propagating');
        setHeights(current_ev, temp_geometry, 'evanescent');

        let line = new Line2( temp_geometry, gridline_material );
        line.computeLineDistances();
        line.scale.set( 1, 1, 1 );
        line.translateY(ybase + ytranslation);
        scene.add( line );
        TVGridlines.push(line);
    }

    //#endregion

    //#region Longitudinal Gridlines

    const LGridlines = [];

    for (let i = 0; i < NLgridlines; i++){
        let xbase = x0s[i * x0step];

        let temp_guided = [];
        let temp_prop = [];
        let temp_ev = []
        for (let j = 0; j < y0s.length; j++){
            let cgs = guided_bytes.slice(Ndata_per_point * (i * x0step + j * x0s.length),
                Ndata_per_point * (i * x0step + j * x0s.length + 1))
            temp_guided.push(cgs[0], cgs[1]);
            let cps = propagating_bytes.slice(Ndata_per_point * (i * x0step + j * x0s.length),
                Ndata_per_point * (i * x0step + j * x0s.length + 1))
            temp_prop.push(cps[0], cps[1]);
            let ces = evanescent_bytes.slice(Ndata_per_point * (i * x0step + j * x0s.length),
                Ndata_per_point * (i * x0step + j * x0s.length + 1))
            temp_ev.push(ces[0], ces[1]);
        }

        const temp_geometry = new LineGeometry();
        temp_geometry.setPositions( LGLBasePositions );
        setHeights(new Float32Array(temp_guided), temp_geometry, 'guided');
        setHeights(new Float32Array(temp_prop), temp_geometry, 'propagating');
        setHeights(new Float32Array(temp_ev), temp_geometry, 'evanescent');

        let line = new Line2( temp_geometry, gridline_material );
        line.computeLineDistances();
        line.scale.set( 1, 1, 1 );
        line.translateX(xbase);
        line.translateY(ytranslation);
        scene.add( line );
        LGridlines.push(line);
    }
    //#endregion

    //#endregion
    
    //#region Border

    const border_points_front = [];
    border_points_front.push( xmin, ytranslation, 0.0 );
    border_points_front.push( xmax, ytranslation, 0.0 );

    const border_points_left = [];
    border_points_left.push( xmin, ytranslation, 0.0 );
    border_points_left.push( xmin, ytranslation + (ymax-ymin), 0.0 );
    
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
    border_line_back.translateY(ymax-ymin);
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
    
    //#endregion

    //#region Interface Lines 
    
    const interface_points = [];
    interface_points.push( Rhos[1], ytranslation, 0.0 );
    interface_points.push( Rhos[1], ymax-ymin+ ytranslation, 0.0 );
    
    const interface_line_geo = new LineGeometry();
    interface_line_geo.setPositions( interface_points);
    
    let interface_line_material = new LineMaterial( {
    
        color: new THREE.Color('silver'),
        linewidth: 1.25, // in pixels
        dashed: true,
        dashSize:.1,
        gapSize:.1,
    } );

    for (let i = 1; i < Rhos.length-1; i++) {            
        const interface_line = new Line2( 
            interface_line_geo, 
            interface_line_material 
        );
        interface_line.computeLineDistances();
        interface_line.scale.set( 1, 1, 1 );
        interface_line.translateX(Rhos[i]-Rhos[1]);
        scene.add( interface_line );
    }

    //#endregion

    //#region Input Field Line
    
    const f0_points = [];
    
    for ( let i = 0; i < f0s.length; i ++ ) {
        const x = x0s[i];
        const y = ytranslation;
        const z = f0s[i];
        f0_points.push( x, y, z );
    
    }
    
    const inputline_geometry = new LineGeometry();
    inputline_geometry.setPositions( f0_points );
        
    let inputline_material = new LineMaterial( {
        color: new THREE.Color().setRGB (0.20784313725490197, 0.5176470588235295, 0.8941176470588236),
        linewidth: 2, // in pixels
        dashed: false
    } );
    
    inputline_material.uniforms.scale = uniforms.scale;
    inputline_material.uniforms.time = uniforms.time;
    inputline_material.uniforms.speed = uniforms.speed;
    
    inputline_material.onBeforeCompile = function ( shader ) {
    
      shader.vertexShader = 'uniform float scale;\n' + shader.vertexShader;
      shader.vertexShader = 'uniform float time;\n' + shader.vertexShader;
      shader.vertexShader = 'uniform float speed;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        'vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );',
        'vec4 start = modelViewMatrix * vec4( vec3(instanceStart.x,instanceStart.y, scale*cos(speed*time)*instanceStart.z), 1.0 );'
      ).replace(
        'vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );',
        'vec4 end = modelViewMatrix * vec4( vec3(instanceEnd.x,instanceEnd.y, scale*cos(speed*time)*instanceEnd.z), 1.0 );'
      );
    
    inputline_material.userData.shader = shader;
    
    };
    
    const inputline = new Line2( inputline_geometry, inputline_material );
    inputline.computeLineDistances();
    inputline.scale.set( 1, 1, 1 );
    scene.add( inputline );

    //#endregion

    //#region MAIN GUI

    //#region SUB GUI: Set Up
    
    const gui_container = document.createElement('div');
    gui_container.id = 'gui_container';
    gui_container.setAttribute(
      'style',
      'position: absolute; z-index: 2; display:block; right: 15px; top: 0px'
    );
    
    subcontainer.appendChild(gui_container);

    const gui = new GUI({container:gui_container});

    //#endregion

    //#region SUB GUI: View Folder 
    
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
    //#endregion

    //#region SUB GUI: Animation Folder
    
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
    //#endregion

    //#region SUB GUI: Components Folder

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

    const radiationFolder = componentsFolder.addFolder("Radiation")
    
    radiationFolder
      .add(componentsParams, 'propagating_on')
      .name('Propagating')
      .onChange((value) =>{mesh.material.uniforms.propagating_on.value = value;
      });
    
    radiationFolder
      .add(componentsParams, 'evanescent_on')
      .name('Evanescent')
      .onChange((value) =>{mesh.material.uniforms.evanescent_on.value = value;
      });
    
    componentsFolder.close();
    //#endregion

    //#region SUB GUI: Appearence Folder

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

    //#region SUB GUI: SUB Appearence Folder: Main Mesh
    
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
    
    meshAppearenceFolder.add(mesh.material.uniforms.colorscale, 'value', 0.0, 2.0).name('colorscale');
    
    meshAppearenceFolder.add(mesh.material, 'wireframe');

    meshAppearenceFolder.add(wiremesh, 'visible').name('show grid');
    
    meshAppearenceFolder.add(wiremesh.material.uniforms.gridlinescale, 'value', .01, 1).name('grid shade')
    
    const param = {
    'grid linewidth': 1.5,
};
    meshAppearenceFolder.add( param, 'grid linewidth', .01, 10 ).onChange( function ( val ) {
    gridline_material.linewidth = val;
} );
    //#endregion

    //#region SUB GUI: SUB Appearence Folder: Input Line
    
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
    
      inputline_material.linewidth = value;
    
    } );
    
    inputFieldFolder
      .addColor( inputFieldParams, 'color')
      .name('line color')
      .onChange(function (value) {
        const color = new THREE.Color().setRGB(value[0], value[1], value[2]);
        inputline_material.color = color;
    });

    inputFieldFolder.close();
    //#endregion

    appearenceFolder.close();

    //#endregion

    // gui.close();

    //#endregion

    //#region Events
    window.addEventListener( 'resize', ()=> {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize( container.clientWidth, container.clientHeight );
    } );
    
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
    //#endregion
    
    //#region Animate/Render
    
    var requestID;
    // renderer.render(scene, camera);

    function animate() {
        requestID = requestAnimationFrame(animate);
        controls.update();
        stopwatch.run();
        uniforms.time.value = stopwatch.time;
        uniforms.update;
        renderer.render(scene, camera);
    }
    
    animate();
    //#endregion
}

export default { render };
