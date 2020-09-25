#define VK_USE_PLATFORM_WIN32_KHR 1
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <tuple>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;
using std::tuple;
namespace fs = std::filesystem;

const static float Pi = 3.14159265359f;

float DegreesToRadians(float Degrees)
{
	return Degrees * (Pi / 180);
}

float RadiansToDegrees(float Radians)
{
	return Radians * (180 / Pi);
}

struct Vector3
{
	float x, y, z;

	Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

	Vector3 operator-(Vector3& rhs)
	{
		return Vector3(x - rhs.x, y - rhs.y, z - rhs.z);
	}

	Vector3 operator+(Vector3& rhs)
	{
		return Vector3(x + rhs.x, y + rhs.y, z + rhs.z);
	}

	Vector3 operator*(Vector3& rhs)
	{
		return Vector3(x * rhs.x, y * rhs.y, z * rhs.z);
	}

	Vector3 operator/(Vector3& rhs)
	{
		return Vector3(x / rhs.x, y / rhs.y, z / rhs.z);
	}

	Vector3 operator/(float rhs)
	{
		return Vector3(x / rhs, y / rhs, z / rhs);
	}


	float Magnitude()
	{
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	}

	Vector3 Normalize()
	{
		return (*this) / Magnitude();
	}

	Vector3 Cross(Vector3& rhs)
	{
		return Vector3((y * rhs.z) - (z * rhs.y), (z * rhs.x) - (x * rhs.z), (x * rhs.y) - (y * rhs.x));
	}

	float Dot(Vector3& rhs)
	{
		return (x * rhs.x) +
			(y * rhs.y) +
			(z * rhs.z);
	}
};

struct Vector4
{
	union
	{
		struct
		{
			/// <summary>
			/// Floats x, y, z, w. Can be aliased as r, g, b ,a.
			/// </summary>
			float x, y, z, w;
		};
		struct
		{
			/// <summary>
			/// Floats r, g, b, a. Can be aliased as x, y, z, w.
			/// </summary>
			float r, g, b, a;
		};
	};

	Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

	Vector4 operator-(Vector4& rhs)
	{
		return Vector4(x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w);
	}

	Vector4 operator+(Vector4& rhs)
	{
		return Vector4(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w);
	}

	Vector4 operator/(Vector4& rhs)
	{
		return Vector4(x / rhs.x, y / rhs.y, z / rhs.z, w / rhs.w);
	}
	Vector4 operator/(float rhs)
	{
		return Vector4(x / rhs, y / rhs, z / rhs, w / rhs);
	}

	float Magnitude()
	{
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(w, 2));
	}

	Vector4 Normalise()
	{
		return (*this) / Magnitude();
	}

	float Dot(Vector4& rhs)
	{
		return (x * rhs.x) +
			(y * rhs.y) +
			(z * rhs.z) +
			(w * rhs.w);
	}

	Vector4 Cross(Vector4& rhs)
	{
		return rhs;
	}
};

float Dot(Vector4 lhs, Vector4 rhs)
{
	return (lhs.x * rhs.x) +
		(lhs.y * rhs.y) +
		(lhs.z * rhs.z) +
		(lhs.w * rhs.w);
}

float Dot(Vector3 lhs, Vector3 rhs)
{
	return (lhs.x * rhs.x) +
		(lhs.y * rhs.y) +
		(lhs.z * rhs.z);
}

struct Matrix4x4
{
	float xa, ya, za, wa;
	float xr, yr, zr, wr;
	float xg, yg, zg, wg;
	float xb, yb, zb, wb;

	Matrix4x4(
		float xa, float ya, float za, float wa,
		float xr, float yr, float zr, float wr,
		float xg, float yg, float zg, float wg,
		float xb, float yb, float zb, float wb) :
		xa(xa), ya(ya), za(za), wa(wa),
		xr(xr), yr(yr), zr(zr), wr(wr),
		xg(xg), yg(yg), zg(zg), wg(wg),
		xb(xb), yb(yb), zb(zb), wb(wb) {}

	Matrix4x4(float init) :
		xa(init), ya(init), za(init), wa(init),
		xr(init), yr(init), zr(init), wr(init),
		xg(init), yg(init), zg(init), wg(init),
		xb(init), yb(init), zb(init), wb(init) {}

	Matrix4x4 operator*(Matrix4x4& rhs)
	{
		Matrix4x4(
			Dot(Vector4(xa, ya, za, wa), Vector4(rhs.xa, rhs.xr, rhs.xg, rhs.xb)),
			Dot(Vector4(xa, ya, za, wa), Vector4(rhs.ya, rhs.yr, rhs.yg, rhs.yb)),
			Dot(Vector4(xa, ya, za, wa), Vector4(rhs.za, rhs.zr, rhs.zg, rhs.zb)),
			Dot(Vector4(xa, ya, za, wa), Vector4(rhs.wa, rhs.wr, rhs.wg, rhs.wb)),

			Dot(Vector4(xr, yr, zr, wr), Vector4(rhs.xa, rhs.xr, rhs.xg, rhs.xb)),
			Dot(Vector4(xr, yr, zr, wr), Vector4(rhs.ya, rhs.yr, rhs.yg, rhs.yb)),
			Dot(Vector4(xr, yr, zr, wr), Vector4(rhs.za, rhs.zr, rhs.zg, rhs.zb)),
			Dot(Vector4(xr, yr, zr, wr), Vector4(rhs.wa, rhs.wr, rhs.wg, rhs.wb)),

			Dot(Vector4(xg, yg, zg, wg), Vector4(rhs.xa, rhs.xr, rhs.xg, rhs.xb)),
			Dot(Vector4(xg, yg, zg, wg), Vector4(rhs.ya, rhs.yr, rhs.yg, rhs.yb)),
			Dot(Vector4(xg, yg, zg, wg), Vector4(rhs.za, rhs.zr, rhs.zg, rhs.zb)),
			Dot(Vector4(xg, yg, zg, wg), Vector4(rhs.wa, rhs.wr, rhs.wg, rhs.wb)),

			Dot(Vector4(xb, yb, zb, wb), Vector4(rhs.xa, rhs.xr, rhs.xg, rhs.xb)),
			Dot(Vector4(xb, yb, zb, wb), Vector4(rhs.ya, rhs.yr, rhs.yg, rhs.yb)),
			Dot(Vector4(xb, yb, zb, wb), Vector4(rhs.za, rhs.zr, rhs.zg, rhs.zb)),
			Dot(Vector4(xb, yb, zb, wb), Vector4(rhs.wa, rhs.wr, rhs.wg, rhs.wb))
		);
	}

	static const Matrix4x4 Identity()
	{
		return Matrix4x4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
	}
};


Matrix4x4 GeneratePerspective(float FoVRadians, float Width, float Height, float NearPlane, float FarPlane);

Matrix4x4 GenerateView(Vector3 CameraPosition, Vector3 CameraTarget, Vector3 CameraUp);

uint32_t getMemTypeIndex(const vk::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);

//Define vertex with position and colour;
struct Vertex
{
	Vector3 pos;
	Vector4 colour;
	Vertex(Vector3 pos, Vector4 col) : pos(pos), colour(col) {}

};

struct SimpleMesh
{
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	SimpleMesh(std::vector<Vertex> vertices, std::vector<uint16_t> indices) : vertices(vertices), indices(indices) {}
};

//Define our triangle in object space x = [-1,1], y = [-1,1]
vector<Vertex> Triangle =
{
	Vertex(Vector3(0.0, -0.5, 0.0), Vector4(1.0, 0.0, 0.0, 1.0)),
	Vertex(Vector3(0.5, 0.5, 0.0), Vector4(0.0, 1.0, 0.0, 1.0)),
	Vertex(Vector3(-0.5, 0.5, 0.0), Vector4(0.0, 0.0, 1.0, 0.0)),
};

SimpleMesh simpleRect = SimpleMesh(
	{
		Vertex(Vector3(-0.5, -0.5, 0.0), Vector4(1.0, 1.0, 0.0, 1.0)),
		Vertex(Vector3(0.5, -0.5, 0.0), Vector4(0.0, 1.0, 0.0, 1.0)),
		Vertex(Vector3(0.5, 0.5, 0.0), Vector4(0.0, 0.0, 1.0, 1.0)),
		Vertex(Vector3(-0.5, 0.5, 0.0), Vector4(1.0, 0.0, 1.0, 1.0))
	}, 
	{
		0, 1, 2, 2, 3, 0
	});

HWND CreateAppWindow(HINSTANCE instance);
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

std::string GetVkMemTypeStr(vk::MemoryPropertyFlags MemPropFlag);
tuple<vk::Buffer, vk::DeviceMemory> createBuffer(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags propertyFlags);

vk::Fence copyBuffer(const vk::Device& device, const vk::Queue& commandQueue, const vk::CommandPool& commandPool, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);

vector<char> LoadShader(string);
vk::ShaderModule CreateModuleForShader(vk::Device& device, vector<char>& shaderData);

int main(int argc, char** argv)
{
	HINSTANCE currInstance = GetModuleHandle(NULL);
	HWND AppWnd = NULL;

	vector<const char*> surfaceExtensionNames = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
	vector<const char*> ValidationLayers = { "VK_LAYER_KHRONOS_validation" };

	auto LayerProps = vk::enumerateInstanceLayerProperties();

	vk::ApplicationInfo appInfo = vk::ApplicationInfo()
		.setApiVersion(VK_API_VERSION_1_1)
		.setApplicationVersion(1)
		.setEngineVersion(1)
		.setPApplicationName("Triangle Test")
		.setPEngineName("Triangle Test Engine");

	vk::InstanceCreateInfo instanceInfo = vk::InstanceCreateInfo()
		.setPApplicationInfo(&appInfo)
		.setEnabledLayerCount(ValidationLayers.size())
		.setPpEnabledLayerNames(ValidationLayers.data())
		.setEnabledExtensionCount(surfaceExtensionNames.size())
		.setPpEnabledExtensionNames(surfaceExtensionNames.data());

	vk::Instance vulkanInstance = vk::createInstance(instanceInfo);
	auto devices = vulkanInstance.enumeratePhysicalDevices();

	if (devices.size() < 1)
	{
		return 1;
	}

	for (auto device : devices)
	{
		//cout << device.getProperties().deviceName << endl;
	}

	vk::PhysicalDevice vulkanPhyDevice = devices[0];

	vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	int index = 0;
	int gfxQueueIndex = 0;

	for (const auto& queues : vulkanPhyDevice.getQueueFamilyProperties())
	{
		float* queuePriorities = new float[queues.queueCount]{ 1.0f };
		queueCreateInfos.push_back(
			vk::DeviceQueueCreateInfo()
			.setQueueFamilyIndex(index)
			.setQueueCount(queues.queueCount)
			.setPQueuePriorities(queuePriorities)
		);
		if (queues.queueFlags & vk::QueueFlagBits::eGraphics)
		{
			gfxQueueIndex = index;
		}

		index++;
	}

	auto phyExt = vulkanPhyDevice.enumerateDeviceExtensionProperties();
	vector<char*> phyDevExtNames = {};

	//Push all Khronos official extensions to device extensions array
	for (int index = 0; index < phyExt.size(); index++)
	{
		std::string ExtNameStr = std::string(phyExt[index].extensionName);
		if (ExtNameStr.find("VK_KHR") != ExtNameStr.npos)
		{
			char* Name = { phyExt[index].extensionName };
			phyDevExtNames.push_back(Name);
		}
	}

	//Allocate device creation info structure and populate
	vk::DeviceCreateInfo deviceCreInfo =
		vk::DeviceCreateInfo()
		.setQueueCreateInfoCount(queueCreateInfos.size())
		.setPQueueCreateInfos(queueCreateInfos.data())
		.setEnabledExtensionCount(phyDevExtNames.size())
		.setPpEnabledExtensionNames(phyDevExtNames.data())
		.setEnabledLayerCount(ValidationLayers.size())
		.setPpEnabledLayerNames(ValidationLayers.data())
		.setPEnabledFeatures(&(vulkanPhyDevice.getFeatures()));

	//Create virtual device interface
	vk::Device vulkanDevice = vulkanPhyDevice.createDevice(deviceCreInfo);
	vk::Queue gfxQueue = vulkanDevice.getQueue(gfxQueueIndex, 0);
	vector<const char*> deviceExtensionNames = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	AppWnd = CreateAppWindow(currInstance);

	if (AppWnd == NULL)
	{
		cout << GetLastError();
		return 1;
	}
	vk::Win32SurfaceCreateInfoKHR surfaceCreateInfo =
		vk::Win32SurfaceCreateInfoKHR()
		.setHinstance(currInstance)
		.setHwnd(AppWnd);

	vk::SurfaceKHR vulkanSurface = vulkanInstance.createWin32SurfaceKHR(surfaceCreateInfo);

	index = 0;
	vector<uint32_t> presentGfxQueueIndex = {};

	for (const auto& queues : vulkanPhyDevice.getQueueFamilyProperties())
	{
		vk::Bool32 supportsPresent = vulkanPhyDevice.getSurfaceSupportKHR(index, vulkanSurface);
		if ((queues.queueFlags & vk::QueueFlagBits::eGraphics) && supportsPresent)
			presentGfxQueueIndex.push_back(index);

		index++;
	}

	auto surfaceCapabilities = vulkanPhyDevice.getSurfaceCapabilitiesKHR(vulkanSurface);
	auto surfacePresentModes = vulkanPhyDevice.getSurfacePresentModesKHR(vulkanSurface);

	vk::CompositeAlphaFlagBitsKHR compositeAlpha =
		(surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
		? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
		: (surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
		? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
		: (surfaceCapabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eInherit)
		? vk::CompositeAlphaFlagBitsKHR::eInherit
		: vk::CompositeAlphaFlagBitsKHR::eOpaque;

	vk::SwapchainCreateInfoKHR swapchainCreateInfo =
		vk::SwapchainCreateInfoKHR()
		.setSurface(vulkanSurface)
		.setImageFormat(vk::Format::eB8G8R8A8Unorm)
		.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
		.setMinImageCount(surfaceCapabilities.minImageCount)
		.setCompositeAlpha(compositeAlpha)
		.setImageExtent(surfaceCapabilities.minImageExtent)
		.setPreTransform(surfaceCapabilities.currentTransform)
		.setPresentMode(surfacePresentModes[3])
		.setQueueFamilyIndexCount(presentGfxQueueIndex.size())
		.setPQueueFamilyIndices(presentGfxQueueIndex.data())
		.setFlags(vk::SwapchainCreateFlagsKHR())
		.setImageArrayLayers(surfaceCapabilities.maxImageArrayLayers)
		.setImageUsage(surfaceCapabilities.supportedUsageFlags)
		.setOldSwapchain(nullptr);

	vk::Queue presentQueue = vulkanDevice.getQueue(presentGfxQueueIndex[0], 0);
	vk::SwapchainKHR vulkanSwapchain;

	try
	{
		vulkanSwapchain = vulkanDevice.createSwapchainKHR(swapchainCreateInfo);
	}
	catch (...)
	{

	}
	auto swapChainImages = vulkanDevice.getSwapchainImagesKHR(vulkanSwapchain);

	vk::ImageCreateInfo depthBufferInfo =
		vk::ImageCreateInfo()
		.setImageType(vk::ImageType::e2D)
		.setFormat(vk::Format::eD16Unorm)
		.setExtent(vk::Extent3D(surfaceCapabilities.minImageExtent, 1))
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
		.setSharingMode(vk::SharingMode::eExclusive);

	auto swapChainImageViews = vector<vk::ImageView>();

	for (auto image : swapChainImages) {
		vk::ImageViewCreateInfo ivCreateInfo =
			vk::ImageViewCreateInfo()
			.setImage(image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(vk::Format::eB8G8R8A8Unorm)
			.setComponents(vk::ComponentMapping()
				.setA(vk::ComponentSwizzle::eIdentity)
				.setR(vk::ComponentSwizzle::eIdentity)
				.setG(vk::ComponentSwizzle::eIdentity)
				.setB(vk::ComponentSwizzle::eIdentity))
			.setSubresourceRange(vk::ImageSubresourceRange()
				.setAspectMask(vk::ImageAspectFlagBits::eColor)
				.setBaseMipLevel(0)
				.setLevelCount(1)
				.setBaseArrayLayer(0)
				.setLayerCount(1));

		auto iv = vulkanDevice.createImageView(ivCreateInfo);
		swapChainImageViews.push_back(iv);
	}

	vk::Image depthBuffer = vulkanDevice.createImage(depthBufferInfo);

	vk::MemoryRequirements depthBufferMemReqs = vulkanDevice.getImageMemoryRequirements(depthBuffer);
	auto ReqedMemFlags = vk::MemoryPropertyFlags(depthBufferMemReqs.memoryTypeBits);

	//Depth buffer seems to need host coherent memory
	auto HostCoherent = ReqedMemFlags & vk::MemoryPropertyFlagBits::eHostCoherent;
	auto DeviceUncachedAMD = ReqedMemFlags & vk::MemoryPropertyFlagBits::eDeviceUncachedAMD;

	auto memProps = vulkanPhyDevice.getMemoryProperties();

	int deviceLocalMemIdx = 0;

	for (int memIdx = 0; memIdx < memProps.memoryTypeCount; memIdx++)
	{
		auto memProp = memProps.memoryTypes[memIdx];
		cout << "Memory type: " << memIdx << endl
			<< "Properties: " << endl << GetVkMemTypeStr(memProp.propertyFlags) << endl
			<< "Heap Offset: " << memProp.heapIndex << endl;

		if (memProp.propertyFlags & ReqedMemFlags)
		{
			deviceLocalMemIdx = memIdx + 1; // Add 1, because the device meme indices aren't zero based :'(
			cout << "Mem Idx: " << deviceLocalMemIdx << endl;
			cout << "_____________________________________" << endl;
			break;
		}
		cout << "_____________________________________" << endl;

	}

	vk::MemoryAllocateInfo depthBufferAllocateInfo =
		vk::MemoryAllocateInfo()
		.setAllocationSize(depthBufferMemReqs.size)
		.setMemoryTypeIndex(deviceLocalMemIdx - 2);

	vk::DeviceMemory depthBufferPtr = vulkanDevice.allocateMemory(depthBufferAllocateInfo);

	vulkanDevice.bindImageMemory(depthBuffer, depthBufferPtr, 0);

	vk::ImageViewCreateInfo imageViewInfo =
		vk::ImageViewCreateInfo()
		.setImage(depthBuffer)
		.setFormat(vk::Format::eD16Unorm)
		.setComponents(vk::ComponentMapping()
			.setA(vk::ComponentSwizzle::eA)
			.setR(vk::ComponentSwizzle::eR)
			.setG(vk::ComponentSwizzle::eG)
			.setB(vk::ComponentSwizzle::eB))
		.setSubresourceRange(vk::ImageSubresourceRange()
			.setAspectMask(vk::ImageAspectFlagBits::eDepth)
			.setBaseMipLevel(0)
			.setLevelCount(1)
			.setBaseArrayLayer(0)
			.setLayerCount(1))
		.setViewType(vk::ImageViewType::e2D);

	vk::ImageView depthBufferView = vulkanDevice.createImageView(imageViewInfo);

	//Load shader programs
	auto vertShader = LoadShader("./shaders/vertex.spv");
	auto fragShader = LoadShader("./shaders/pixel.spv");

	auto vertShaderModule = CreateModuleForShader(vulkanDevice, vertShader);
	auto fragShaderModule = CreateModuleForShader(vulkanDevice, fragShader);

	//Setup pipeline stages
	auto pipelineShaderStages = vector<vk::PipelineShaderStageCreateInfo>{
		vk::PipelineShaderStageCreateInfo()
		.setStage(vk::ShaderStageFlagBits::eVertex)
		.setModule(vertShaderModule)
		.setPName("main"),
		vk::PipelineShaderStageCreateInfo()
		.setStage(vk::ShaderStageFlagBits::eFragment)
		.setModule(fragShaderModule)
		.setPName("main")
	};

	vk::VertexInputBindingDescription vertBindingDesc =
		vk::VertexInputBindingDescription()
		.setBinding(0)
		.setStride(sizeof(Vertex))
		.setInputRate(vk::VertexInputRate::eVertex);

	vk::VertexInputAttributeDescription vertexInputAttrs[] = {
		vk::VertexInputAttributeDescription()
		.setBinding(0)
		.setLocation(0)
		.setFormat(vk::Format::eR32G32B32Sfloat)
		.setOffset(offsetof(Vertex, Vertex::pos)),
		vk::VertexInputAttributeDescription()
		.setBinding(0)
		.setLocation(1)
		.setFormat(vk::Format::eR32G32B32A32Sfloat)
		.setOffset(offsetof(Vertex, Vertex::colour))
	};

	vk::PipelineVertexInputStateCreateInfo vertexInputCreateInfo =
		vk::PipelineVertexInputStateCreateInfo()
		.setVertexBindingDescriptionCount(1)
		.setPVertexBindingDescriptions(&vertBindingDesc)
		.setVertexAttributeDescriptionCount(2)
		.setPVertexAttributeDescriptions(vertexInputAttrs);

	vk::PipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo =
		vk::PipelineInputAssemblyStateCreateInfo()
		.setTopology(vk::PrimitiveTopology::eTriangleList)
		.setPrimitiveRestartEnable(false);

	vk::Viewport viewport = vk::Viewport()
		.setX(0.0f)
		.setY(0.0f)
		.setWidth(surfaceCapabilities.minImageExtent.width)
		.setHeight(surfaceCapabilities.minImageExtent.height)
		.setMinDepth(0.0f)
		.setMaxDepth(1.0f);

	vk::Rect2D  scissor = vk::Rect2D().setExtent(surfaceCapabilities.minImageExtent).setOffset({ 0, 0 });

	vk::PipelineViewportStateCreateInfo viewportStateCreateInfo =
		vk::PipelineViewportStateCreateInfo()
		.setViewportCount(1)
		.setPViewports(&viewport)
		.setScissorCount(1)
		.setPScissors(&scissor);

	vk::PipelineRasterizationStateCreateInfo rasterizerStateCreateInfo =
		vk::PipelineRasterizationStateCreateInfo()
		.setDepthClampEnable(false)
		.setRasterizerDiscardEnable(false)
		.setPolygonMode(vk::PolygonMode::eFill)
		.setLineWidth(1.0f)
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setFrontFace(vk::FrontFace::eClockwise)
		.setDepthBiasEnable(false)
		.setDepthBiasConstantFactor(0.0f)
		.setDepthBiasClamp(0.0f)
		.setDepthBiasSlopeFactor(0.0f);

	vk::PipelineMultisampleStateCreateInfo multisampleStateCreateInfo =
		vk::PipelineMultisampleStateCreateInfo()
		.setSampleShadingEnable(false)
		.setRasterizationSamples(vk::SampleCountFlagBits::e1)
		.setMinSampleShading(1.0f)
		.setAlphaToCoverageEnable(false)
		.setAlphaToOneEnable(false);

	vk::PipelineColorBlendAttachmentState colourBlendAttachmentDesc =
		vk::PipelineColorBlendAttachmentState()
		.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
		.setBlendEnable(false)
		.setSrcColorBlendFactor(vk::BlendFactor::eOne)
		.setDstColorBlendFactor(vk::BlendFactor::eZero)
		.setColorBlendOp(vk::BlendOp::eAdd)
		.setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
		.setDstAlphaBlendFactor(vk::BlendFactor::eZero)
		.setAlphaBlendOp(vk::BlendOp::eAdd);

	vk::PipelineColorBlendStateCreateInfo colourBlendStateCreateInfo =
		vk::PipelineColorBlendStateCreateInfo()
		.setLogicOpEnable(false)
		.setAttachmentCount(1)
		.setPAttachments(&colourBlendAttachmentDesc);

	vk::DescriptorSetLayoutBinding vulkanDescriptorSetLayoutBinding =
		vk::DescriptorSetLayoutBinding()
		.setBinding(0)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(1)
		.setStageFlags(vk::ShaderStageFlagBits::eVertex);

	vk::DescriptorSetLayoutCreateInfo vulkanDescSetLayoutInfo =
		vk::DescriptorSetLayoutCreateInfo()
		.setBindingCount(1)
		.setPBindings(&vulkanDescriptorSetLayoutBinding);

	vk::DescriptorSetLayout vulkanDescriptorSetLayout = vulkanDevice.createDescriptorSetLayout(vulkanDescSetLayoutInfo);

	vk::PipelineLayoutCreateInfo vulkanPipelineLayoutInfo =
		vk::PipelineLayoutCreateInfo()
		.setSetLayoutCount(1)
		.setPSetLayouts(&vulkanDescriptorSetLayout)
		.setPushConstantRangeCount(0);

	vk::PipelineLayout vulkanPipelineLayout = vulkanDevice.createPipelineLayout(vulkanPipelineLayoutInfo);

	vk::AttachmentDescription colourAttachment =
		vk::AttachmentDescription()
		.setFormat(vk::Format::eB8G8R8A8Unorm)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		.setStoreOp(vk::AttachmentStoreOp::eStore)
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	vk::AttachmentReference colourAttachmentRef =
		vk::AttachmentReference()
		.setAttachment(0)
		.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

	vk::SubpassDescription subpassDesc =
		vk::SubpassDescription()
		.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics) // This is where you could switch for Compute or RayTraceNV
		.setColorAttachmentCount(1)
		.setPColorAttachments(&colourAttachmentRef);

	vk::SubpassDependency spDependency =
		vk::SubpassDependency()
		.setSrcSubpass(VK_SUBPASS_EXTERNAL)
		.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
		.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
		.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

	vk::RenderPassCreateInfo renderPassCreateInfo =
		vk::RenderPassCreateInfo()
		.setAttachmentCount(1)
		.setPAttachments(&colourAttachment)
		.setSubpassCount(1)
		.setPSubpasses(&subpassDesc)
		.setDependencyCount(1)
		.setPDependencies(&spDependency);

	vk::RenderPass renderPass = vulkanDevice.createRenderPass(renderPassCreateInfo);

	vk::GraphicsPipelineCreateInfo graphicsPipelineInfo =
		vk::GraphicsPipelineCreateInfo()
		.setStageCount(2)
		.setPStages(pipelineShaderStages.data())
		.setPVertexInputState(&vertexInputCreateInfo)
		.setPInputAssemblyState(&inputAssemblyCreateInfo)
		.setPViewportState(&viewportStateCreateInfo)
		.setPRasterizationState(&rasterizerStateCreateInfo)
		.setPMultisampleState(&multisampleStateCreateInfo)
		.setPDepthStencilState(nullptr)
		.setPColorBlendState(&colourBlendStateCreateInfo)
		.setPDynamicState(nullptr)
		.setLayout(vulkanPipelineLayout)
		.setRenderPass(renderPass)
		.setSubpass(0);

	vk::Pipeline vulkanGraphicsPipeline = vulkanDevice.createGraphicsPipeline(vk::PipelineCache(), graphicsPipelineInfo);

	//Setup render Passes
	cout << "SC Image View Count: " << swapChainImageViews.size() << endl;
	vector<vk::Framebuffer> swapChainFrameBuffers = vector<vk::Framebuffer>();

	for (auto imageView : swapChainImageViews) {
		vk::FramebufferCreateInfo fbCreateInfo =
			vk::FramebufferCreateInfo()
			.setRenderPass(renderPass)
			.setAttachmentCount(1)
			.setPAttachments(&imageView)
			.setWidth(surfaceCapabilities.minImageExtent.width)
			.setHeight(surfaceCapabilities.minImageExtent.height)
			.setLayers(1);

		swapChainFrameBuffers.push_back(vulkanDevice.createFramebuffer(fbCreateInfo));
	}

	//Setup Command pool
	vk::CommandPoolCreateInfo commandPoolInfo =
		vk::CommandPoolCreateInfo()
		.setQueueFamilyIndex(gfxQueueIndex);

	vk::CommandPool vulkanCommandPool = vulkanDevice.createCommandPool(commandPoolInfo);

	vk::CommandBufferAllocateInfo commandBufferInfo =
		vk::CommandBufferAllocateInfo()
		.setCommandBufferCount(swapChainFrameBuffers.size())
		.setCommandPool(vulkanCommandPool)
		.setLevel(vk::CommandBufferLevel::ePrimary);

	vector<vk::CommandBuffer> vulkanCommandBuffersArray = vulkanDevice.allocateCommandBuffers(commandBufferInfo);

	Matrix4x4 ProjectionMatrix = GeneratePerspective(90.0f, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.minImageExtent.height, 1.0f, 100.0f);
	Matrix4x4 ViewMatrix = GenerateView(Vector3(-5.0f, 3.0f, -10.0f), Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, -1.0f, 0.0f));
	Matrix4x4 ClipMatrix = Matrix4x4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, -1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.5f, 0.0f,
		0.0f, 0.0f, 0.5f, 1.0f
	);

	//Setting up a vertex buffer for consumption by draw, using a host 

	vk::DeviceSize meshVertSize = sizeof(Vertex) * simpleRect.vertices.size();
	vk::DeviceSize meshIndicesSize = sizeof(uint16_t) * simpleRect.indices.size();
	auto [vertStagingBuffer, vertStagingBufferMemory] = createBuffer(vulkanDevice, vulkanPhyDevice,
		meshVertSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	void* vertStagingMemoryPtr;
	vulkanDevice.mapMemory(vertStagingBufferMemory, 0, meshVertSize, vk::MemoryMapFlagBits(0), &vertStagingMemoryPtr);
	memcpy(vertStagingMemoryPtr, simpleRect.vertices.data(), (size_t)meshVertSize);
	vulkanDevice.unmapMemory(vertStagingBufferMemory);

	auto [vertexBuffer, vertexBufferMemory] = createBuffer(vulkanDevice, vulkanPhyDevice,
		meshVertSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);

	auto [indexStagingBuffer, indexStagingBufferMemory] = createBuffer(vulkanDevice, vulkanPhyDevice,
		meshIndicesSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	vector<vk::Fence> resourceFences;

	void* indStagingMemoryPtr;
	vulkanDevice.mapMemory(indexStagingBufferMemory, 0, meshVertSize, vk::MemoryMapFlagBits(0), &indStagingMemoryPtr);
	memcpy(indStagingMemoryPtr, simpleRect.indices.data(), (size_t)meshIndicesSize);
	vulkanDevice.unmapMemory(indexStagingBufferMemory);

	auto [indexBuffer, indexBufferMemory] = createBuffer(vulkanDevice, vulkanPhyDevice,
		meshVertSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);

	resourceFences.push_back(copyBuffer(vulkanDevice, gfxQueue, vulkanCommandPool, vertStagingBuffer, vertexBuffer, meshVertSize));
	resourceFences.push_back(copyBuffer(vulkanDevice, gfxQueue, vulkanCommandPool, indexStagingBuffer, indexBuffer, meshIndicesSize));

	vulkanDevice.waitForFences(resourceFences, true, 1000000);

	//Record render sequence to command buffer
	for (int i = 0; i < vulkanCommandBuffersArray.size(); i++) {
		auto buffer = vulkanCommandBuffersArray[i];
		vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo();

		buffer.begin(beginInfo);

		vk::ClearValue clearColour = vk::ClearValue()
			.setColor(vk::ClearColorValue(std::array<float, 4Ui64>({ (100.0f / 255.0f), (149.0f / 255.0f), (237.0f / 255.0f), 1.0f }))); // Cornflower, the DirectX classic blue normalized to 0.0 - 1.0

		vk::RenderPassBeginInfo renderPassBeginInfo =
			vk::RenderPassBeginInfo()
			.setRenderPass(renderPass)
			.setFramebuffer(swapChainFrameBuffers[i])
			.setRenderArea(vk::Rect2D().setOffset({ 0,0 }).setExtent(surfaceCapabilities.minImageExtent))
			.setClearValueCount(1)
			.setPClearValues(&clearColour);

		buffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, vulkanGraphicsPipeline);

		vk::ArrayProxy<const vk::Buffer> vertexBuffers = vk::ArrayProxy<const vk::Buffer>(1, &vertexBuffer);
		vk::DeviceSize offsets[] = { 0 };
		vk::ArrayProxy<const vk::DeviceSize> offsetsProxy = vk::ArrayProxy<const vk::DeviceSize>(1, offsets);

		buffer.bindVertexBuffers(0, vertexBuffers, offsetsProxy);
		buffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);

		buffer.drawIndexed(simpleRect.indices.size(), 1, 0, 0, 0);

		buffer.endRenderPass();

		buffer.end();
	}

	//Setup default semaphores
	vk::SemaphoreCreateInfo semaphoreCreateInfo = vk::SemaphoreCreateInfo();
	vk::Semaphore imageAvailableSemaphore = vulkanDevice.createSemaphore(semaphoreCreateInfo);
	vk::Semaphore renderFinishedSemaphore = vulkanDevice.createSemaphore(semaphoreCreateInfo);

	MSG message = { 0 };
	while (message.message != WM_QUIT)
	{
		while (PeekMessage(&message, 0, 0, 0, PM_REMOVE))
		{
			if (message.message == WM_QUIT)
				return 0;

			TranslateMessage(&message);
			DispatchMessage(&message);
		}

		//The vulkan draw loop
		uint32_t imageIndex;
		vulkanDevice.acquireNextImageKHR(vulkanSwapchain, UINT64_MAX, imageAvailableSemaphore, vk::Fence(), &imageIndex);

		vk::Semaphore waitSemaphores[] = { imageAvailableSemaphore };
		vk::Semaphore signalSemaphores[] = { renderFinishedSemaphore };
		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
		vk::SubmitInfo submitInfo =
			vk::SubmitInfo()
			.setWaitSemaphoreCount(1)
			.setPWaitSemaphores(waitSemaphores)
			.setPWaitDstStageMask(waitStages)
			.setCommandBufferCount(1)
			.setPCommandBuffers(&vulkanCommandBuffersArray[imageIndex])
			.setSignalSemaphoreCount(1)
			.setPSignalSemaphores(signalSemaphores);

		gfxQueue.submit(submitInfo, vk::Fence());
		vk::SwapchainKHR swapChains[] = { vulkanSwapchain };

		vk::PresentInfoKHR presentInfo =
			vk::PresentInfoKHR()
			.setWaitSemaphoreCount(1)
			.setPWaitSemaphores(signalSemaphores)
			.setSwapchainCount(1)
			.setPSwapchains(swapChains)
			.setPImageIndices(&imageIndex);

		presentQueue.presentKHR(presentInfo);

		vulkanDevice.waitIdle();
	}

	vulkanDevice.destroyBuffer(vertexBuffer);

	return 0;
}

Matrix4x4 GeneratePerspective(float FoVDegrees, float Width, float Height, float NearPlane, float FarPlane)
{
	float FoVRadians = DegreesToRadians(FoVDegrees);

	//Projection focal length = 1 / tan(FOV / 2) with FOV in radians
	float FocalLength = 1.0f / tan(FoVRadians / 2.0f);

	float AspectRatio = Height / Width;
	float TanA = tan(FoVRadians / 2);

	//Vertical FOV (Radians) = 2arctan(AspectRatio / FocalLength)
	float VerticalFoV = 2 * atan(AspectRatio / FocalLength);

	float Near2 = 2 * NearPlane;
	float NearFar2 = 2 * NearPlane * FarPlane;

	return Matrix4x4(
		(1.0f / (TanA * AspectRatio)), 0.0f, 0.0f, 0.0f,
		0.0f, (1.0f / TanA), 0.0f, 0.0f,
		0.0f, 0.0f, -(FarPlane + NearPlane / FarPlane - NearPlane), -1.0f,
		0.0f, 0.0f, ((-NearFar2) / FarPlane - NearPlane), 0.0f);
}

Matrix4x4 GenerateView(Vector3 CameraPosition, Vector3 CameraTarget, Vector3 CameraUp)
{
	Vector3 zAxis = (CameraPosition - CameraTarget).Normalize();
	Vector3 xAxis = CameraUp.Cross(zAxis).Normalize();
	Vector3 yAxis = zAxis.Cross(xAxis);

	Matrix4x4 ViewMatrix = Matrix4x4
	(
		xAxis.x, yAxis.x, zAxis.x, 0,
		xAxis.y, yAxis.y, zAxis.y, 0,
		xAxis.z, yAxis.z, zAxis.z, 0,
		-xAxis.Dot(CameraPosition), -yAxis.Dot(CameraPosition), -zAxis.Dot(CameraPosition), 1
	);


	return ViewMatrix;
}

uint32_t getMemTypeIndex(const vk::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw "Unable to find usable device memory for buffer";
}

HWND CreateAppWindow(HINSTANCE instance)
{
	WNDCLASSEX wc = {};
	wc.cbSize = sizeof(wc);
	wc.hInstance = instance;
	wc.lpfnWndProc = WindowProc;
	wc.lpszClassName = L"VulkanTest";

	if (!RegisterClassEx(&wc))
	{

		return 0;
	}

	HWND hwnd = CreateWindowEx(0, L"VulkanTest", L"Vulkan Triangle",
		WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, NULL, NULL, instance, NULL);

	ShowWindow(hwnd, SW_SHOW);

	return hwnd;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		return 0;
	}
	}

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

std::string GetVkMemTypeStr(vk::MemoryPropertyFlags MemPropFlag)
{
	std::stringstream out;

	out << vk::to_string(MemPropFlag);

	return out.str();
}

tuple<vk::Buffer, vk::DeviceMemory> createBuffer(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, vk::DeviceSize size, vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags propertyFlags)
{
	vk::BufferCreateInfo bufferCreateInfo =
		vk::BufferCreateInfo()
		.setSize(size)
		.setUsage(usageFlags)
		.setSharingMode(vk::SharingMode::eExclusive);

	vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

	vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(buffer);

	vk::MemoryAllocateInfo mAllocInfo =
		vk::MemoryAllocateInfo()
		.setAllocationSize(memReqs.size)
		.setMemoryTypeIndex(getMemTypeIndex(physicalDevice, memReqs.memoryTypeBits, propertyFlags));

	vk::DeviceMemory bufferMemory = device.allocateMemory(mAllocInfo);

	device.bindBufferMemory(buffer, bufferMemory, 0);

	return tuple(buffer, bufferMemory);
}

vk::Fence copyBuffer(const vk::Device& device, const vk::Queue& commandQueue, const vk::CommandPool& commandPool, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
	vk::CommandBufferAllocateInfo allocInfo =
		vk::CommandBufferAllocateInfo()
		.setLevel(vk::CommandBufferLevel::ePrimary)
		.setCommandPool(commandPool)
		.setCommandBufferCount(1);

	vector<vk::CommandBuffer> copyCommandBufferResult = device.allocateCommandBuffers(allocInfo);

	if (copyCommandBufferResult.size() != 1) {
		device.freeCommandBuffers(commandPool, copyCommandBufferResult);
		throw std::runtime_error("Incorrect number of command buffers created, something has gone wierd");
	}

	vk::CommandBuffer copyCommandBuffer = copyCommandBufferResult[0];

	vk::CommandBufferBeginInfo copyBeginInfo =
		vk::CommandBufferBeginInfo()
		.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

	copyCommandBuffer.begin(copyBeginInfo);

	vk::BufferCopy copyMemInfo =
		vk::BufferCopy()
		.setSrcOffset(0)
		.setDstOffset(0)
		.setSize(size);

	copyCommandBuffer.copyBuffer(srcBuffer, dstBuffer, copyMemInfo);

	copyCommandBuffer.end();

	vk::SubmitInfo copySubmitInfo =
		vk::SubmitInfo()
		.setCommandBufferCount(1)
		.setPCommandBuffers(&copyCommandBuffer);

	vk::FenceCreateInfo fenceInfo =
		vk::FenceCreateInfo();

	vk::Fence copyFence = device.createFence(fenceInfo);

	commandQueue.submit(copySubmitInfo, copyFence);

	return copyFence;
}

vector<char> LoadShader(string shaderPath)
{
	fs::path shaderLoc = fs::absolute(shaderPath);
	if (fs::exists(shaderLoc)) {
		auto shaderSize = fs::file_size(shaderLoc);
		vector<char> shaderData = vector<char>(shaderSize);
		std::ifstream shaderFileStream = std::ifstream(shaderLoc, std::ifstream::in | std::ifstream::binary);
		shaderFileStream.read(shaderData.data(), shaderSize);
		return shaderData;
	}
	else {
		throw std::exception("An error occured reading shader data.");
	}
}

vk::ShaderModule CreateModuleForShader(vk::Device& device, vector<char>& shaderData)
{
	vk::ShaderModuleCreateInfo createInfo =
		vk::ShaderModuleCreateInfo()
		.setCodeSize(shaderData.size())
		.setPCode(reinterpret_cast<uint32_t*>(shaderData.data()));

	return device.createShaderModule(createInfo);
}
