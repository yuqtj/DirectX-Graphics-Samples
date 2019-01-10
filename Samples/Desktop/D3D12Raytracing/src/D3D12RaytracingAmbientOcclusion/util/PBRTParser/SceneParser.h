//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

//
// Original implementation: https://github.com/wallisc/DuosRenderer/tree/DXRRenderer/PBRTParser
//

#pragma once

namespace SceneParser
{

	struct Vector2
	{
		union {
			struct {
				float x, y;
			};
			struct {
				float u, v;
			};
			DirectX::XMFLOAT2 xmFloat2;
		};

		float &operator[](UINT i)
		{
			switch (i)
			{
			case 0:
				return x;
			case 1:
				return y;
			default:
				assert(false);
				return y;
			}
		}

		XMVECTOR ToXMVECTOR() { return XMLoadFloat2(&xmFloat2); }
	};

	struct Vector3
	{
		Vector3(float nX, float nY, float nZ) : x(nX), y(nY), z(nZ) {}
		Vector3() : Vector3(0, 0, 0) {}

		float &operator[](UINT i)
		{
			switch (i)
			{
			case 0:
				return x;
			case 1:
				return y;
			case 2:
				return z;
			default:
				assert(false);
				return z;
			}
		}
		union {
			struct {
				float x, y, z;
			};
			struct {
				float r, g, b;
			};
			DirectX::XMFLOAT3 xmFloat3;
		};
		XMVECTOR GetXMVECTOR() { return XMLoadFloat3(&xmFloat3); }
	};

	struct Film
	{
		UINT m_ResolutionX;
		UINT m_ResolutionY;
		std::string m_Filename;
	};

	struct Camera
	{
		// In Degrees. The is the narrower of the view frustrums width/height
		float m_FieldOfView;
		float m_NearPlane;
		float m_FarPlane;
		Vector3 m_Position;
		Vector3 m_LookAt;
		Vector3 m_Up;
	};

	struct Material
	{
		std::string m_MaterialName;
		Vector3 m_Diffuse;
		Vector3 m_Specular;
		Vector3 m_Opacity;
		float m_URoughness;
		float m_VRoughness;
		std::string m_DiffuseTextureFilename;
		std::string m_SpecularTextureFilename;
		std::string m_OpacityTextureFilename;
        std::string m_NormalMapTextureFilename;
	};

	struct Vertex
	{
		Vector3 Normal;
		Vector3 Position;
		Vector3 Tangents;
		Vector2 UV;
	};

	typedef UINT Index;

    struct Mesh
    {
        Material *m_pMaterial;
        std::vector<Index> m_IndexBuffer;
        std::vector<Vertex> m_VertexBuffer;
		XMMATRIX m_transform;

        void GenerateTangents()
        {
            // Zero tangents.
            for (auto& vertex : m_VertexBuffer)
            {
                vertex.Tangents.xmFloat3 = XMFLOAT3(0, 0, 0);
            }

            // Add tangents from all triangles a vertex corresponds to.
            for (UINT i = 0; i < m_IndexBuffer.size(); i += 3)
            {
                auto& index0 = m_IndexBuffer[i];
                auto& index1 = m_IndexBuffer[i+1];
                auto& index2 = m_IndexBuffer[i+2];
                XMFLOAT3& v0 = m_VertexBuffer[index0].Position.xmFloat3;
                XMFLOAT3& v1 = m_VertexBuffer[index1].Position.xmFloat3;
                XMFLOAT3& v2 = m_VertexBuffer[index2].Position.xmFloat3;

                XMFLOAT2& uv0 = m_VertexBuffer[index0].UV.xmFloat2;
                XMFLOAT2& uv1 = m_VertexBuffer[index1].UV.xmFloat2;
                XMFLOAT2& uv2 = m_VertexBuffer[index2].UV.xmFloat2;

                Vector3& tangent1 = m_VertexBuffer[index0].Tangents;
                Vector3& tangent2 = m_VertexBuffer[index1].Tangents;
                Vector3& tangent3 = m_VertexBuffer[index2].Tangents;

                XMVECTOR tangent = XMLoadFloat3(&CalculateTangent(v0, v1, v2, uv0, uv1, uv2));

                XMStoreFloat3(&tangent1.xmFloat3, tangent1.GetXMVECTOR() + tangent);
                XMStoreFloat3(&tangent2.xmFloat3, tangent2.GetXMVECTOR() + tangent);
                XMStoreFloat3(&tangent3.xmFloat3, tangent3.GetXMVECTOR() + tangent);
            }

            // Renormalize the tangents.
            for (auto& vertex : m_VertexBuffer)
            {
                XMStoreFloat3(&vertex.Tangents.xmFloat3, XMVector3Normalize(XMLoadFloat3(&vertex.Tangents.xmFloat3)));
            }
        }
    };

    struct AreaLight
    {
        AreaLight(Vector3 LightColor) : m_LightColor(LightColor) {}

        Mesh m_Mesh;
        Vector3 m_LightColor;
    };

    struct EnvironmentMap
    {
        EnvironmentMap() {}
        EnvironmentMap(const std::string &fileName) : m_FileName(fileName) {}
        std::string m_FileName;
    };

    struct Scene
    {
        Camera m_Camera;
        Film m_Film;
        std::unordered_map<std::string, Material> m_Materials;
        std::vector<AreaLight> m_AreaLights;
        std::vector<Mesh> m_Meshes;
        EnvironmentMap m_EnvironmentMap;
		XMMATRIX m_transform;
    };

    class BadFormatException : public std::exception
    {
    public:
        BadFormatException(char const* const errorMessage) : std::exception(errorMessage) {}
    };

    class SceneParserClass
    {
        virtual void Parse(std::string filename, Scene &outputScene, bool bClockwiseWindingORder = true, bool rhCoords = false) = 0;
    };
};
