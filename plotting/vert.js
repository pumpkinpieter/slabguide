export default `
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
        
        float zpos = (cos(speed*time)*Real + sin(speed*time)*Imag);
        
        result = vec4( position.x, position.y, scale * zpos, 1.0 );
        int index = int(floor(clamp(255.0/colorscale*(zpos-colorscale*colormin)/(colormax - colormin), 0.0, 255.0)));
        vColor = vLut[index];
        gl_Position = projectionMatrix * modelViewMatrix * result;
    }`
;