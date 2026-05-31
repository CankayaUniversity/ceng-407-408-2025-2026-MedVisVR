// HUD uzerine ZTest Always ile cizilen 2D slice shader'i.
// SliceRenderingShader'in kopyasi + her seyin ustunde cizecek sekilde
// render state'leri degistirildi. Volume objesi HUD'u artik kapatamaz.
Shader "VolumeRendering/HUD/SliceRender2DHUD"
{
    Properties
    {
        _DataTex("Data Texture (Generated)", 3D) = "" {}
        _TFTex("Transfer Function Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "Queue" = "Overlay" "RenderPipeline" = "UniversalPipeline" "IgnoreProjector" = "True" }
        LOD 100

        Pass
        {
            Cull Off
            ZTest Always
            ZWrite Off
            Blend Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // Quest 3 multiview / single-pass instanced icin stereo varyantlari
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct appdata
            {
                UNITY_VERTEX_INPUT_INSTANCE_ID
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                UNITY_VERTEX_OUTPUT_STEREO
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 relVert : TEXCOORD1;
            };

            Texture3D _DataTex;             SamplerState sampler_DataTex;
            Texture2D _TFTex;               SamplerState sampler_TFTex;

            uniform float4x4 _parentInverseMat;
            uniform float4x4 _planeMat;

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                float3 worldPos = TransformObjectToWorld(v.vertex.xyz);
                o.vertex = TransformWorldToHClip(worldPos);
                float2 uvMod = float2((0.5f - v.uv.x) * 10.0f, (0.5f - v.uv.y) * 10.0f);
                float3 sampleWorld = mul(_planeMat, float4(uvMod.x, 0.0f, uvMod.y, 1.0f));
                o.relVert = mul(_parentInverseMat, float4(sampleWorld, 1.0f));
                o.uv = v.uv;
                return o;
            }

            half4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float3 dataCoord = i.relVert + float3(0.5f, 0.5f, 0.5f);
                if (dataCoord.x > 1.0f || dataCoord.y > 1.0f || dataCoord.z > 1.0f ||
                    dataCoord.x < 0.0f || dataCoord.y < 0.0f || dataCoord.z < 0.0f)
                {
                    return half4(0.02f, 0.02f, 0.02f, 1.0f);
                }
                float dataVal = _DataTex.Sample(sampler_DataTex, dataCoord);
                half4 col = _TFTex.Sample(sampler_TFTex, float2(dataVal, 0.0));
                col.a = 1.0f;
                return col;
            }
            ENDHLSL
        }
    }

    // Built-in RP fallback (Editor/Oyun icinde URP yoksa)
    SubShader
    {
        Tags { "Queue" = "Overlay" "IgnoreProjector" = "True" }
        LOD 100

        Pass
        {
            Cull Off
            ZTest Always
            ZWrite Off
            Blend Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float4 relVert : TEXCOORD1;
            };

            sampler3D _DataTex;
            sampler2D _TFTex;
            uniform float4x4 _parentInverseMat;
            uniform float4x4 _planeMat;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                float2 uvMod = float2((0.5f - v.uv.x) * 10.0f, (0.5f - v.uv.y) * 10.0f);
                float3 sampleWorld = mul(_planeMat, float4(uvMod.x, 0.0f, uvMod.y, 1.0f));
                o.relVert = mul(_parentInverseMat, float4(sampleWorld, 1.0f));
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float3 dataCoord = i.relVert + float3(0.5f, 0.5f, 0.5f);
                if (dataCoord.x > 1.0f || dataCoord.y > 1.0f || dataCoord.z > 1.0f ||
                    dataCoord.x < 0.0f || dataCoord.y < 0.0f || dataCoord.z < 0.0f)
                {
                    return fixed4(0.02f, 0.02f, 0.02f, 1.0f);
                }
                float dataVal = tex3D(_DataTex, dataCoord).r;
                fixed4 col = tex2D(_TFTex, float2(dataVal, 0.0));
                col.a = 1.0f;
                return col;
            }
            ENDCG
        }
    }
}
